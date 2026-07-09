"""웹 앱 비동기 실행 래퍼 테스트.

무거운 워크플로우 콜백은 trame 이벤트 루프를 막지 않도록 비동기 래퍼
(`_run_async`)가 동기 워커를 executor 에서 실행하고, 진행 플래그(`nt_busy`)를
켰다 끄며, GL 렌더는 메인 스레드로 미룬다. 여기서는 GL 없이(`build_ui=False`,
plotter=None → 렌더 no-op) 그 계약을 검증한다.
"""

from __future__ import annotations

import pytest

pytest.importorskip("trame", reason="비동기 래퍼 테스트에는 trame 이 필요합니다.")
pytest.importorskip("pyvista", reason="비동기 래퍼 테스트에는 pyvista 가 필요합니다.")


def _make_app(name: str):
    from trame.app import get_server

    from naviertwin.web.app import NavierTwinWebApp

    return NavierTwinWebApp(server=get_server(name, client_type="vue3"))


@pytest.mark.asyncio
async def test_run_async_sets_and_clears_busy() -> None:
    app = _make_app("nt-async-busy")
    st = app.server.state
    app.load_demo()
    assert st.nt_busy is False

    await app._run_async(app.run_pod, "POD 계산 중…")
    # 워커 완료 후 busy 해제, 결과 state 반영, 오류 없음.
    assert st.nt_pod_done is True
    assert st.nt_busy is False
    assert st.nt_error == ""


@pytest.mark.asyncio
async def test_run_async_defers_render_then_flushes() -> None:
    app = _make_app("nt-async-render")
    st = app.server.state
    app.load_demo()
    st.nt_method = "q_criterion"

    # render_after=True 면 워커 중에는 렌더가 보류되고, 복귀 후 메인 스레드에서 flush.
    await app._run_async(app.run_analysis, "분석 중…", render_after=True)
    assert st.nt_error == ""
    assert "Q-criterion" in st.nt_fields
    # 워커 종료 후 defer 플래그/보류 렌더가 정리되어야 한다.
    assert app._defer_render is False
    assert app._render_pending is False


@pytest.mark.asyncio
async def test_run_async_reports_worker_error() -> None:
    app = _make_app("nt-async-err")
    st = app.server.state
    # 데이터 없이 POD → 워커가 _fail 로 오류 보고, busy 는 해제.
    await app._run_async(app.run_pod, "POD 계산 중…")
    assert st.nt_busy is False
    assert st.nt_error


@pytest.mark.asyncio
async def test_bench_train_async_streams_progress() -> None:
    """라이브 학습: 모니터 큐가 epoch/진행률/손실 시리즈를 상태에 스트리밍한다."""
    import asyncio

    app = _make_app("nt-async-live")
    st = app.server.state
    st.nt_bench_kind = "heat"
    st.nt_bench_nsamples = 10
    st.nt_bench_nx = 24
    app.bench_generate()
    st.nt_bench_epochs = 12
    st.nt_bench_modes = 8
    st.nt_bench_width = 12

    max_progress = -1.0
    max_series = 0
    task = asyncio.ensure_future(app._bench_train_async())
    while not task.done():
        await asyncio.sleep(0.03)
        max_progress = max(max_progress, float(st.nt_progress))
        max_series = max(max_series, len(st.nt_bench_loss_series))
    await task

    assert st.nt_error == ""
    assert st.nt_bench_trained is True
    assert st.nt_bench_training is False
    assert max_progress > 0  # 학습 중 진행률이 갱신됨
    assert max_series >= 2  # 손실 시리즈가 실시간으로 자람
    assert st.nt_bench_train_summary
