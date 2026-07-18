"""형상 인지(geometry-aware) FNO — SDF/마스크/운전조건 채널 입력.

형상이 케이스마다 다른 정상(steady) 케이스 세트를 **공통 균일 격자** 위에서
학습하는 서로게이트. 형상은 부호거리(SDF)·유체 마스크 채널로, 운전조건은 상수
브로드캐스트 채널로 입력에 들어간다 — DeepCFD(Ribeiro et al., 2020)·Thuerey
et al.(2020) 계열의 표준 인코딩이다. 샘플 하나 = 케이스 하나(시간축 없음).

입력 텐서는 :func:`naviertwin.core.operator_learning.fno.case_tensorizer.
cases_to_grid_tensors` 가 만든다: 채널 = ``[sdf, mask, μ_1, ..., μ_k]``.

주의(소표본): 문헌의 데이터 규모는 수백~수천 케이스다(DeepCFD ~1000개 형상).
이 클래스는 5~10 케이스의 소표본에서도 **동작은 하도록**(크래시 없이 학습·추론)
설계했지만, 그 결과는 정량 예측이 아니라 정성적 데모로 해석해야 한다.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.operator_learning.fno.geometry_fno import GeometryFNO2D
    >>> N, H, W = 5, 24, 32
    >>> X = np.random.rand(N, H, W, 3).astype(np.float32)  # [sdf, mask, mu_0]
    >>> Y = np.random.rand(N, H, W, 1).astype(np.float32)  # [p]
    >>> op = GeometryFNO2D(n_params=1, modes=6, width=8, epochs=2)
    >>> op.fit(X, Y)                              # doctest: +SKIP
    >>> op.predict(X).shape                       # doctest: +SKIP
    (5, 24, 32, 1)
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)

#: 지원하는 FNO 백엔드 — "builtin"(자체 FNO2D) | "neuralop"(레퍼런스 구현).
GEOMETRY_FNO_BACKENDS = ("builtin", "neuralop")


class GeometryFNO2D:
    """SDF/마스크/운전조건 채널을 입력받는 2D 형상 인지 FNO 서로게이트.

    기존 FNO 기계(:class:`~naviertwin.core.operator_learning.fno.fno.FNO2D`
    또는 :class:`~naviertwin.core.operator_learning.fno.neuralop_fno.
    NeuralOpFNO`)를 그대로 재사용하고, 이 클래스는 그 위에서

    - 채널 계약 검증 (입력 = 2 + n_params 채널, 출력 = out_channels 채널),
    - 채널별 표준화/역표준화 (평균·표준편차 저장),
    - ndarray 직접 입출력 인터페이스 (``fit(inputs, targets)``)

    만 담당한다. ``fit(dict)`` 계약의 :class:`BaseOperator` 와 시그니처가
    달라(배열 2개 직접 전달) 상속 대신 합성(composition)을 쓴다.

    Attributes:
        n_params: 운전조건 파라미터 수 (입력 채널 = 2 + n_params).
        in_channels: 입력 채널 수 (= 2 + n_params).
        out_channels: 출력(타깃 필드) 채널 수.
        backend: 사용 중인 FNO 백엔드 ("builtin" | "neuralop").
        train_losses_: epoch 별 평균 학습 손실 (표준화 공간의 MSE).
        is_fitted: :meth:`fit` 완료 여부.
        input_mean_ / input_std_: 입력 채널별 표준화 통계. shape = (C_in,).
        output_mean_ / output_std_: 출력 채널별 표준화 통계. shape = (C_out,).
        training_metadata: 학습 요약 (v5.6) — ``device_used``("cuda:0"/"cpu"),
            ``use_amp``, ``batch_size``, ``backend`` 등. UI 디바이스 배지가 읽는다.
    """

    def __init__(
        self,
        n_params: int,
        out_channels: int = 1,
        modes: int = 12,
        width: int = 32,
        n_layers: int = 4,
        epochs: int = 200,
        lr: float = 1e-3,
        device: str = "auto",
        seed: int | None = 0,
        backend: str = "builtin",
        epoch_callback: Optional[Callable[[int, float], None]] = None,
        batch_size: int | None = None,
        use_amp: bool = False,
    ) -> None:
        """초기화.

        Args:
            n_params: 운전조건 파라미터 수 (k ≥ 0). 입력 채널 = 2 + k.
            out_channels: 출력 필드 채널 수.
            modes: 축별 유지 푸리에 모드 수 (두 축 동일 적용).
            width: 은닉 채널 폭.
            n_layers: 푸리에 층 수.
            epochs: 학습 epoch 수 (Adam + MSE).
            lr: Adam 학습률.
            device: "auto" | "cpu" | "cuda" 등. "neuralop" 백엔드는 자체
                규칙(cuda 가능 시 cuda)을 따르므로 이 값이 무시된다.
            seed: 난수 시드. None 이면 고정하지 않는다.
            backend: "builtin"(자체 FNO2D) | "neuralop"(레퍼런스 FNO).
            epoch_callback: ``(epoch_idx, epoch_loss)`` 라이브 진행 콜백.
            batch_size: 미니배치 크기 (v5.6). ``None`` 이면 백엔드 기본값을
                그대로 쓴다 — 기존 동작과 동일.
            use_amp: AMP(자동 혼합정밀) 사용 (v5.6). builtin 백엔드에서 CUDA
                일 때만 켜지고 CPU 는 조용한 no-op. "neuralop" 백엔드는 자체
                학습 루프가 AMP 를 지원하지 않아 무시된다.

        Raises:
            ValueError: ``n_params`` 가 음수이거나 ``backend`` 가 지원 목록에
                없는 경우.
        """
        if int(n_params) < 0:
            raise ValueError(f"n_params 는 0 이상이어야 합니다. 현재: {n_params}")
        if backend not in GEOMETRY_FNO_BACKENDS:
            raise ValueError(
                f"지원하지 않는 backend: {backend!r}. 지원: {list(GEOMETRY_FNO_BACKENDS)}"
            )
        self.n_params = int(n_params)
        self.out_channels = int(out_channels)
        self.modes = int(modes)
        self.width = int(width)
        self.n_layers = int(n_layers)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.device = device
        self.seed = seed
        self.backend = backend
        self.epoch_callback = epoch_callback
        self.batch_size = None if batch_size is None else int(batch_size)
        self.use_amp = bool(use_amp)

        self.is_fitted: bool = False
        self.train_losses_: list[float] = []
        self.training_metadata: dict[str, object] = {}
        self.input_mean_: NDArray[np.float32] | None = None
        self.input_std_: NDArray[np.float32] | None = None
        self.output_mean_: NDArray[np.float32] | None = None
        self.output_std_: NDArray[np.float32] | None = None
        self._model: Any = None

    @property
    def in_channels(self) -> int:
        """입력 채널 수 (= sdf + mask + 파라미터 k개)."""
        return 2 + self.n_params

    @property
    def device_used(self) -> str | None:
        """학습에 실제 사용된 torch 디바이스 문자열 ("cuda:0"/"cpu").

        :meth:`fit` 전에는 ``None``. ``training_metadata["device_used"]`` 와
        같은 값이다 (UI 디바이스 배지용).
        """
        value = self.training_metadata.get("device_used")
        return str(value) if value is not None else None

    # ------------------------------------------------------------------
    # 내부 유틸
    # ------------------------------------------------------------------

    @staticmethod
    def _channel_stats(
        values: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """(N, H, W, C) 텐서의 채널별 평균/표준편차 (상수 채널은 std=1)."""
        mean = values.mean(axis=(0, 1, 2)).astype(np.float32)
        std = values.std(axis=(0, 1, 2)).astype(np.float32)
        std = np.where(std < 1e-8, np.float32(1.0), std)
        return mean, std

    @staticmethod
    def _masked_channel_stats(
        values: NDArray[np.float32],
        mask: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """유효셀에서만 계산한 채널별 평균/표준편차.

        0-채움된 고체/도메인 밖 셀을 통계에서 제외해 표준화가 자리채움 0 에
        끌려가지 않게 한다 (외부 리뷰: 0 은 물리값이 아니다). 유효셀이 하나도
        없으면 전체 셀 통계로 안전하게 되돌린다.

        Args:
            values: (N, H, W, C) 타깃 텐서.
            mask: (N, H, W) 유효셀 지시자 (1 유효 / 0 자리채움).

        Returns:
            ``(mean, std)`` — 각각 shape (C,), 상수 채널은 std=1.
        """
        valid = mask.reshape(-1) > 0.5
        flat = values.reshape(-1, values.shape[-1])
        if not valid.any():
            return GeometryFNO2D._channel_stats(values)
        sel = flat[valid]
        mean = sel.mean(axis=0).astype(np.float32)
        std = sel.std(axis=0).astype(np.float32)
        std = np.where(std < 1e-8, np.float32(1.0), std)
        return mean, std

    def _build_model(self) -> Any:
        """설정에 맞는 백엔드 FNO 인스턴스를 만든다.

        ``batch_size=None`` 이면 백엔드 자체 기본값을 유지한다 (기존 동작).
        ``use_amp`` 는 builtin(FNO2D)에만 전달된다 — neuralop 래퍼의 학습
        루프는 AMP 미지원이라 조용히 무시한다.
        """
        if self.backend == "neuralop":
            from naviertwin.core.operator_learning.fno.neuralop_fno import NeuralOpFNO

            extra: dict[str, Any] = {}
            if self.batch_size is not None:
                extra["batch_size"] = self.batch_size
            if self.use_amp:
                logger.debug("GeometryFNO2D: neuralop 백엔드는 use_amp 를 무시합니다.")
            return NeuralOpFNO(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                modes=self.modes,
                width=self.width,
                n_dim=2,
                n_layers=self.n_layers,
                max_epochs=self.epochs,
                lr=self.lr,
                seed=self.seed,
                epoch_callback=self.epoch_callback,
                **extra,
            )
        from naviertwin.core.operator_learning.fno.fno import FNO2D

        extra = {}
        if self.batch_size is not None:
            extra["batch_size"] = self.batch_size
        return FNO2D(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            modes1=self.modes,
            modes2=self.modes,
            width=self.width,
            n_layers=self.n_layers,
            max_epochs=self.epochs,
            lr=self.lr,
            device=self.device,
            seed=self.seed,
            epoch_callback=self.epoch_callback,
            use_amp=self.use_amp,
            **extra,
        )

    def _validate_inputs(self, values: NDArray[np.float32], what: str) -> None:
        if values.ndim != 4:
            raise ValueError(
                f"{what} 는 (N, H, W, C) 4D 텐서여야 합니다. 현재: {values.shape}"
            )

    def _resolve_sample_masks(
        self,
        x: NDArray[np.float32],
        sample_masks: NDArray[np.float32] | None,
        mask_from_channel: bool,
    ) -> NDArray[np.float32] | None:
        """손실 마스크를 (N, H, W) float32 로 정규화한다 (없으면 None).

        ``sample_masks`` 가 우선하며, 없고 ``mask_from_channel`` 이 참이면
        ``inputs[..., 1]`` (mask 입력 채널)에서 유도한다.

        Raises:
            ValueError: 마스크 shape 이 (N, H, W)/(N, H, W, 1) 이 아니거나
                inputs 의 (N, H, W) 와 다른 경우.
        """
        if sample_masks is None:
            if not mask_from_channel:
                return None
            # mask 입력 채널(index 1)에서 자동 유도.
            return np.ascontiguousarray(x[..., 1], dtype=np.float32)

        mask = np.asarray(sample_masks, dtype=np.float32)
        if mask.ndim == 4 and mask.shape[-1] == 1:
            mask = mask[..., 0]
        if mask.ndim != 3:
            raise ValueError(
                f"sample_masks 는 (N, H, W) 또는 (N, H, W, 1) 여야 합니다. "
                f"현재: {mask.shape}"
            )
        if mask.shape != x.shape[:3]:
            raise ValueError(
                f"sample_masks 의 (N, H, W) 가 inputs 와 다릅니다: "
                f"{mask.shape} vs {x.shape[:3]}"
            )
        return np.ascontiguousarray(mask, dtype=np.float32)

    def _fit_builtin_masked(
        self,
        x_std: NDArray[np.float32],
        y_std: NDArray[np.float32],
        mask: NDArray[np.float32],
    ) -> Any:
        """builtin FNO2D 를 마스킹 MSE 로 학습한다 (fno.py 를 건드리지 않음).

        :class:`~naviertwin.core.operator_learning.fno.fno.FNO2D` 의 모델 구성
        (:meth:`FNO2D._build`)·디바이스 해석·AMP 규칙·:meth:`FNO2D.predict` 를
        그대로 재사용하되, 학습 루프만 여기서 돌려 손실을

            L = Σ(m · (pred − target)²) / Σ_broadcast(m)

        로 바꾼다. m 을 출력 채널로 브로드캐스트해 합산 분모를 잡으므로 마스크가
        전부 1 이면 ``torch.nn.MSELoss`` (전체 평균)와 값·기울기 모두 비트 단위로
        같다 — 빠른 경로와의 동치성이 보장된다. 분모는 유효셀이 하나도 없을 때만
        작동하는 ``clamp_min(eps)`` 로 0-나눗셈을 막는다.

        모델 구성/셔플 RNG 소비 순서를 :meth:`FNO2D.fit` 과 동일하게 맞춰,
        같은 시드에서 전부-1 마스크는 무마스크 학습과 비트 단위로 일치한다.

        Returns:
            학습이 끝난 ``FNO2D`` 인스턴스 (``predict`` 가능 상태).
        """
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        fno = self._build_model()  # FNO2D 인스턴스 (builtin backend)
        # FNO2D.fit 과 동일한 순서로 디바이스 해석 → 모델 구성(시드 소비 동일).
        fno._device = fno._resolve_device()
        fno._model = fno._build().to(fno._device)
        optim = torch.optim.Adam(fno._model.parameters(), lr=fno.lr)

        # AMP 규칙도 FNO2D.fit 과 동일 (CPU 는 조용한 no-op → 수치 불변).
        amp_on = bool(fno.use_amp) and fno._device.type == "cuda"
        if amp_on and not torch.cuda.is_bf16_supported():
            logger.warning("FNO2D: GPU 가 bfloat16 을 지원하지 않아 AMP 를 끕니다.")
            amp_on = False
        fno._amp_used = amp_on

        X = np.asarray(x_std, dtype=np.float32)
        Y = np.asarray(y_std, dtype=np.float32)
        M = np.asarray(mask, dtype=np.float32)[..., None]  # (N, H, W, 1)
        eps = 1e-8

        loader = DataLoader(
            TensorDataset(torch.tensor(X), torch.tensor(Y), torch.tensor(M)),
            batch_size=min(fno.batch_size, len(X)),
            shuffle=True,
        )
        fno.train_losses_ = []
        epoch_idx = 0
        while epoch_idx < fno.max_epochs:
            epoch_loss = 0.0
            batches = iter(loader)
            while True:
                try:
                    xb, yb, mb = next(batches)
                except StopIteration:
                    break
                xb = xb.to(fno._device)
                yb = yb.to(fno._device)
                mb = mb.to(fno._device)
                optim.zero_grad()
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=amp_on
                ):
                    pred = fno._model(xb)
                    sq = (pred - yb) ** 2
                    num = (mb * sq).sum()
                    # m 을 출력 채널로 브로드캐스트한 합산 분모 — 전부-1 이면
                    # 정확히 원소 수라 MSELoss(전체 평균)와 비트 단위로 같다.
                    denom = mb.expand_as(sq).sum()
                    loss = num / denom.clamp_min(eps)
                loss.backward()
                optim.step()
                epoch_loss += float(loss.item()) * xb.shape[0]
            epoch_loss /= max(len(X), 1)
            fno.train_losses_.append(epoch_loss)
            epoch_idx += 1
            if fno.epoch_callback is not None:
                fno.epoch_callback(epoch_idx, epoch_loss)

        fno.n_epochs = fno.max_epochs
        fno.is_fitted = True
        logger.info(
            "GeometryFNO2D 마스킹 학습 완료: loss=%.6g",
            fno.train_losses_[-1] if fno.train_losses_ else float("nan"),
        )
        return fno

    # ------------------------------------------------------------------
    # 학습 / 추론
    # ------------------------------------------------------------------

    def fit(
        self,
        inputs: NDArray[np.float32],
        targets: NDArray[np.float32],
        sample_masks: NDArray[np.float32] | None = None,
        *,
        mask_from_channel: bool = False,
    ) -> None:
        """케이스 텐서로 형상 인지 FNO 를 학습한다 (Adam + MSE).

        입출력을 채널별로 표준화한 뒤 백엔드 FNO 에 위임한다. 소표본
        (N = 5~10)에서도 동작하지만 결과는 정성적 데모 수준이다 — 문헌
        (DeepCFD 등)의 정량 성능은 수백~수천 케이스에서 얻어진다.

        **마스킹 손실 (외부 리뷰 반영):** 타깃의 고체/도메인 밖 격자점은 물리값이
        아니라 자리채움용 0 이다. ``sample_masks`` 가 주어지면 이 영역을 손실에서
        제외한 **마스킹 MSE** 를 쓴다::

            L = Σ(m · (pred − target)²) / (Σ m + eps)

        (m 은 출력 채널 축으로 브로드캐스트되며, eps 는 유효셀이 0 일 때만 작동하는
        0-나눗셈 방지 clamp 다 — 마스크가 전부 1 이면 평범한 MSE 와 **비트 단위로**
        같아 기존 수치를 보존한다.) 나아가 이때 출력 표준화 통계도 **유효셀에서만**
        계산해 자리채움 0 이 정규화를 왜곡하지 않게 한다. 입력 표준화는 종전대로
        전체 셀 기준이다.

        ``sample_masks`` 가 ``None`` 이면 동작은 종전과 완전히 동일하다(빠른 경로 —
        백엔드 ``MSELoss`` 에 위임).

        Args:
            inputs: (N, H, W, 2 + n_params) float32 — [sdf, mask, μ...].
            targets: (N, H, W, out_channels) float32 — 타깃 필드 채널.
                고체/도메인 밖 셀은 0 자리채움이며, 마스킹 손실을 쓰면 점수화되지
                않는다.
            sample_masks: (N, H, W) 또는 (N, H, W, 1) float32 유효셀 마스크
                (1 유효/실데이터, 0 자리채움). 보통
                :func:`~naviertwin.core.operator_learning.fno.case_tensorizer.
                cases_to_grid_tensors` 의 ``"valid_mask"`` 를 그대로 넘긴다.
                ``None`` 이면 마스킹하지 않는다(기존 동작).
            mask_from_channel: ``True`` 이고 ``sample_masks`` 가 ``None`` 이면
                ``inputs[..., 1]`` (mask 입력 채널)에서 마스크를 자동 유도한다.
                기존 수치/테스트 보존을 위해 기본값은 ``False`` 다.

        Raises:
            ValueError: shape/채널 수가 계약과 맞지 않는 경우.
            NotImplementedError: ``sample_masks`` 를 준 상태로 ``backend=
                "neuralop"`` 을 쓴 경우 (neuralop 백엔드는 자체 내부 손실을 써
                마스킹할 수 없다 — builtin 백엔드만 마스킹을 지원한다).
        """
        x = np.asarray(inputs, dtype=np.float32)
        y = np.asarray(targets, dtype=np.float32)
        self._validate_inputs(x, "inputs")
        self._validate_inputs(y, "targets")
        if x.shape[-1] != self.in_channels:
            raise ValueError(
                f"입력 채널 수가 계약(2 + n_params = {self.in_channels})과 "
                f"다릅니다. 현재: {x.shape[-1]} (shape={x.shape})"
            )
        if y.shape[-1] != self.out_channels:
            raise ValueError(
                f"타깃 채널 수가 out_channels({self.out_channels})와 다릅니다. "
                f"현재: {y.shape[-1]} (shape={y.shape})"
            )
        if x.shape[:3] != y.shape[:3]:
            raise ValueError(
                f"inputs/targets 의 (N, H, W) 가 다릅니다: {x.shape[:3]} vs {y.shape[:3]}"
            )
        if x.shape[0] < 1:
            raise ValueError("학습 케이스가 최소 1개 필요합니다.")

        mask = self._resolve_sample_masks(x, sample_masks, mask_from_channel)
        if mask is not None and self.backend == "neuralop":
            raise NotImplementedError(
                "neuralop 백엔드는 자체 내부 손실을 사용해 마스킹 손실을 적용할 수 "
                "없습니다. 마스킹(sample_masks)이 필요하면 backend='builtin' 을 "
                "쓰세요."
            )

        self.input_mean_, self.input_std_ = self._channel_stats(x)
        if mask is not None:
            # 유효셀에서만 출력 통계 계산 — 0-채움이 정규화를 왜곡하지 않게.
            self.output_mean_, self.output_std_ = self._masked_channel_stats(y, mask)
        else:
            self.output_mean_, self.output_std_ = self._channel_stats(y)
        x_std = (x - self.input_mean_) / self.input_std_
        y_std = (y - self.output_mean_) / self.output_std_

        if mask is None:
            # 빠른 경로 — 백엔드 MSELoss 에 위임 (기존 동작과 비트 단위 동일).
            self._model = self._build_model()
            self._model.fit({"inputs": x_std, "outputs": y_std})
        else:
            # 마스킹 경로 — builtin FNO2D 내부를 재사용하되 마스킹 MSE 로 학습.
            self._model = self._fit_builtin_masked(x_std, y_std, mask)
        self.train_losses_ = list(self._model.train_losses_)
        self.is_fitted = True
        # v5.6: 실제 학습 디바이스/가속 설정 보고 — UI 디바이스 배지가 읽는다.
        device_obj = getattr(self._model, "_device", None)
        device_used = str(device_obj) if device_obj is not None else None
        self.training_metadata = {
            "backend": self.backend,
            "device_used": device_used,
            # 백엔드가 보고하는 실효 AMP 상태 (CUDA + bf16 지원일 때만 True;
            # neuralop 백엔드는 AMP 미지원 → False).
            "use_amp": bool(getattr(self._model, "_amp_used", False)),
            "batch_size": self.batch_size,
            "n_cases": int(x.shape[0]),
            "epochs": int(self.epochs),
            "masked_loss": mask is not None,
            "final_loss": (
                float(self.train_losses_[-1]) if self.train_losses_ else None
            ),
        }
        logger.info(
            "GeometryFNO2D 학습 완료: backend=%s, N=%d, in=%d ch, out=%d ch, loss=%.6g",
            self.backend,
            x.shape[0],
            self.in_channels,
            self.out_channels,
            self.train_losses_[-1] if self.train_losses_ else float("nan"),
        )

    def predict(self, inputs: NDArray[np.float32]) -> NDArray[np.float32]:
        """새 형상/운전조건 텐서에 대한 필드를 예측한다.

        Args:
            inputs: (N, H, W, 2 + n_params) 또는 (H, W, 2 + n_params).

        Returns:
            (N, H, W, out_channels) float32. 배치 없는 입력이면 배치 없는
            (H, W, out_channels) 를 돌려준다. 값은 원 물리 스케일로
            역표준화되어 있다.

        Raises:
            RuntimeError: :meth:`fit` 이 호출되지 않은 경우.
            ValueError: 입력 shape/채널 수가 계약과 맞지 않는 경우.
        """
        if not self.is_fitted or self._model is None:
            raise RuntimeError("GeometryFNO2D 의 fit() 을 먼저 호출해야 합니다.")
        x = np.asarray(inputs, dtype=np.float32)
        squeeze = x.ndim == 3
        if squeeze:
            x = x[np.newaxis, ...]
        self._validate_inputs(x, "inputs")
        if x.shape[-1] != self.in_channels:
            raise ValueError(
                f"입력 채널 수가 계약(2 + n_params = {self.in_channels})과 "
                f"다릅니다. 현재: {x.shape[-1]} (shape={x.shape})"
            )
        assert self.input_mean_ is not None and self.input_std_ is not None
        assert self.output_mean_ is not None and self.output_std_ is not None

        x_std = (x - self.input_mean_) / self.input_std_
        y_std = np.asarray(self._model.predict({"x": x_std}), dtype=np.float32)
        y = y_std * self.output_std_ + self.output_mean_
        return y[0] if squeeze else y

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        return (
            f"GeometryFNO2D(backend={self.backend!r}, n_params={self.n_params}, "
            f"out_channels={self.out_channels}, status={status})"
        )


__all__ = ["GEOMETRY_FNO_BACKENDS", "GeometryFNO2D"]
