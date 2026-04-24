"""외부 CFD 솔버 래퍼 모음 + 내장 간이 솔버.

- ``lbm_d2q9`` : 2D D2Q9 Lattice Boltzmann (numpy 직접 구현, optional GPU 없음)
- ``lettuce_wrapper`` : Lettuce 라이브러리 선호
- ``flowtorch_wrapper`` : flowtorch 기반 POD/DMD 파이프라인 연동
- ``jax_fluids_wrapper`` : JAX-Fluids 선호
"""
