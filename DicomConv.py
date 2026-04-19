from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from openpyxl import Workbook


INPUT_FOLDER = r"F:\Datasets\EndoKI\UKER\EndoKI shared DATA\cia839"
OUTPUT_FOLDER = r"F:\Datasets\EndoKI\UKER\dicom converted"

INPUT_BASENAME = Path(INPUT_FOLDER).name
REPORT_XLSX = str(Path(OUTPUT_FOLDER) / f"conversion_report_{INPUT_BASENAME}.xlsx")


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)

    min_val = np.min(arr)
    max_val = np.max(arr)

    if max_val <= min_val:
        return np.zeros(arr.shape, dtype=np.uint8)

    arr = (arr - min_val) / (max_val - min_val)
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return arr


def get_display_array(ds: pydicom.dataset.FileDataset) -> np.ndarray:
    arr = ds.pixel_array

    number_of_frames = int(getattr(ds, "NumberOfFrames", 1) or 1)
    if number_of_frames > 1:
        arr = arr[0]

    try:
        arr = apply_voi_lut(arr, ds)
    except Exception:
        pass

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = arr.astype(np.float32) * slope + intercept

    photometric = str(getattr(ds, "PhotometricInterpretation", "")).upper()
    if photometric == "MONOCHROME1":
        arr = np.max(arr) - arr

    return arr


def make_image(arr: np.ndarray) -> Image.Image:
    if arr.ndim == 2:
        arr8 = normalize_to_uint8(arr)
        return Image.fromarray(arr8, mode="L")

    if arr.ndim == 3:
        if arr.shape[-1] == 3:
            if arr.dtype != np.uint8:
                arr = normalize_to_uint8(arr)
            return Image.fromarray(arr, mode="RGB")

        if arr.shape[0] == 3:
            arr = np.transpose(arr, (1, 2, 0))
            if arr.dtype != np.uint8:
                arr = normalize_to_uint8(arr)
            return Image.fromarray(arr, mode="RGB")

    raise ValueError(f"Unsupported pixel array shape: {arr.shape}")


def get_metadata_for_report(dcm_path: Path) -> dict:
    info = {
        "file_path": str(dcm_path),
        "file_name": dcm_path.name,
        "transfer_syntax_uid": "UNKNOWN",
        "sop_class_uid": "UNKNOWN",
        "photometric_interpretation": "UNKNOWN",
        "rows": "",
        "columns": "",
        "number_of_frames": "",
    }

    try:
        ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True, force=True)

        info["transfer_syntax_uid"] = str(getattr(ds.file_meta, "TransferSyntaxUID", "UNKNOWN"))
        info["sop_class_uid"] = str(getattr(ds, "SOPClassUID", "UNKNOWN"))
        info["photometric_interpretation"] = str(getattr(ds, "PhotometricInterpretation", "UNKNOWN"))
        info["rows"] = str(getattr(ds, "Rows", ""))
        info["columns"] = str(getattr(ds, "Columns", ""))
        info["number_of_frames"] = str(getattr(ds, "NumberOfFrames", ""))

    except Exception:
        pass

    return info


def write_excel_report(report_path: str, failed_rows: list[dict]) -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = "Conversion Report"

    headers = [
        "file_path",
        "file_name",
        "transfer_syntax_uid",
        "sop_class_uid",
        "photometric_interpretation",
        "rows",
        "columns",
        "number_of_frames",
        "error",
    ]
    ws.append(headers)

    for row in failed_rows:
        ws.append([row.get(h, "") for h in headers])

    # Basic column sizing
    for column_cells in ws.columns:
        max_length = 0
        column_letter = column_cells[0].column_letter
        for cell in column_cells:
            value = "" if cell.value is None else str(cell.value)
            if len(value) > max_length:
                max_length = len(value)
        ws.column_dimensions[column_letter].width = min(max_length + 2, 60)

    wb.save(report_path)


def convert_one_dicom(dcm_path: Path, input_root: Path, output_root: Path) -> Optional[Path]:
    ds = pydicom.dcmread(str(dcm_path), force=True)
    arr = get_display_array(ds)
    img = make_image(arr)

    relative_path = dcm_path.relative_to(input_root)
    output_dir = output_root / relative_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{dcm_path.stem}.png"
    img.save(str(output_file), format="PNG", compress_level=0)

    return output_file


def convert_folder(input_folder: str, output_folder: str, report_xlsx: str) -> None:
    input_root = Path(input_folder).resolve()
    output_root = Path(output_folder).resolve()

    if not input_root.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_root}")

    output_root.mkdir(parents=True, exist_ok=True)

    dcm_files = list(input_root.rglob("*.dcm"))

    if not dcm_files:
        print("No .dcm files found.")
        return

    converted = 0
    failed = 0
    failed_rows = []

    for dcm_file in dcm_files:
        try:
            output_file = convert_one_dicom(dcm_file, input_root, output_root)
            print(f"Converted: {dcm_file} -> {output_file}")
            converted += 1

        except Exception as e:
            meta = get_metadata_for_report(dcm_file)
            meta["error"] = str(e)
            failed_rows.append(meta)

            print(f"Failed to convert {dcm_file}")
            print(f"  TransferSyntaxUID: {meta['transfer_syntax_uid']}")
            print(f"  Error: {meta['error']}")
            failed += 1

    write_excel_report(report_xlsx, failed_rows)

    print("\nDone.")
    print(f"Converted: {converted}")
    print(f"Failed:    {failed}")
    print(f"Report:    {report_xlsx}")


if __name__ == "__main__":
    convert_folder(INPUT_FOLDER, OUTPUT_FOLDER, REPORT_XLSX)