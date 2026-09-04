"""Scientific evidence contracts."""

from femx.validation.report import CheckStatus, ValidationCheck, ValidationReport
from femx.validation.tpu_collective_evidence import (
    TpuCollectiveProcessSetEvidence,
    aggregate_tpu_collective_process_evidence,
)
from femx.validation.tpu_distributed_electrothermal_evidence import (
    TpuDistributedElectrothermalProcessSetEvidence,
    aggregate_tpu_distributed_electrothermal_process_evidence,
)
from femx.validation.tpu_distributed_fdtdx_thermo_optic_evidence import (
    TpuDistributedFDTDXThermoOpticProcessSetEvidence,
    aggregate_tpu_distributed_fdtdx_thermo_optic_process_evidence,
)
from femx.validation.tpu_fdtdx_mode_source_evidence import (
    TpuFdtdxModeSourceProcessSetEvidence,
    aggregate_tpu_fdtdx_mode_source_process_evidence,
)
from femx.validation.tpu_fdtdx_waveguide_source_evidence import (
    TpuFdtdxWaveguideSourceProcessSetEvidence,
    aggregate_tpu_fdtdx_waveguide_source_process_evidence,
)
from femx.validation.tpu_public_ring_heater_evidence import (
    TpuPublicRingHeaterProcessSetEvidence,
    aggregate_tpu_public_ring_heater_process_evidence,
)

__all__ = [
    "CheckStatus",
    "TpuCollectiveProcessSetEvidence",
    "TpuDistributedElectrothermalProcessSetEvidence",
    "TpuDistributedFDTDXThermoOpticProcessSetEvidence",
    "TpuFdtdxModeSourceProcessSetEvidence",
    "TpuFdtdxWaveguideSourceProcessSetEvidence",
    "TpuPublicRingHeaterProcessSetEvidence",
    "ValidationCheck",
    "ValidationReport",
    "aggregate_tpu_collective_process_evidence",
    "aggregate_tpu_distributed_electrothermal_process_evidence",
    "aggregate_tpu_distributed_fdtdx_thermo_optic_process_evidence",
    "aggregate_tpu_fdtdx_mode_source_process_evidence",
    "aggregate_tpu_fdtdx_waveguide_source_process_evidence",
    "aggregate_tpu_public_ring_heater_process_evidence",
]
