from dataclasses import dataclass

import pandas as pd

from tdf_simulator.config import RunConfig, TDFConfig


@dataclass
class TDFInfoBuilder:  # noqa: D101
    tdf_config: TDFConfig
    run_config: RunConfig

    def build_global_metadata(self, max_num_peaks: int) -> pd.DataFrame:
        """Builds the global metadata for the TDF file.

        Args:
            max_num_peaks (int): The maximum number of peaks per scan.
                Can be derived from the `frames` table as max(df["NumPeaks"]).

        """
        global_meta_data = {
            #     "SchemaType": "TDF",
            #     "SchemaVersionMajor": 3,
            #     "SchemaVersionMinor": 7,
            #     "AcquisitionSoftwareVendor": "Bruker",
            #     "InstrumentVendor": "Bruker",
            #     "ClosedProperly": 1,
            "TimsCompressionType": 2,
            "MaxNumPeaksPerScan": int(max_num_peaks),
            #     "AnalysisId": "00000000-0000-0000-0000-000000000000",
            "DigitizerNumSamples": self.tdf_config.NUM_TOF_BINS,
            "MzAcqRangeLower": self.tdf_config.MZ_MIN,
            "MzAcqRangeUpper": self.tdf_config.MZ_MAX,
            "AcquisitionSoftware": "timsTOF",
            #     "AcquisitionSoftwareVersion": "0.0",
            #     "AcquisitionFirmwareVersion": "0.1",
            #     "AcquisitionDateTime": "2023-05-05T21:20:37.229+02:00",
            #     "InstrumentName": "timsTOF SCP",
            #     "InstrumentFamily": 9,
            #     "InstrumentRevision": 3,
            #     "InstrumentSourceType": 11,
            #     "InstrumentSerialNumber": 0,
            #     "OperatorName": "Admin",
            #     "Description": "",
            "SampleName": "test",
            #     "MethodName": "test.m",
            #     "DenoisingEnabled": 0,
            #     "PeakWidthEstimateValue": 0.000025,
            #     "PeakWidthEstimateType": 1,
            #     "PeakListIndexScaleFactor": 1,
            "OneOverK0AcqRangeLower": self.tdf_config.IM_MIN,
            "OneOverK0AcqRangeUpper": self.tdf_config.IM_MAX,
            #     "DigitizerType": "SA248P",
            #     "DigitizerSerialNumber": "AQ00074235",
        }
        global_meta_data = pd.DataFrame(
            {
                "Key": global_meta_data.keys(),
                "Value": global_meta_data.values(),
            }
        )
        return global_meta_data
