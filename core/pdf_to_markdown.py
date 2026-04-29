from __future__ import annotations

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import EasyOcrOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


def _build_converter() -> DocumentConverter:
    """构建支持OCR和公式识别的PDF文档转换器。"""
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.ocr_options = EasyOcrOptions(lang=["en", "ch_sim"])
    pipeline_options.do_formula_enrichment = True
    pipeline_options.images_scale = 2.0

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            )
        }
    )


def pdf_to_markdown(pdf_path: str) -> str:
    """将PDF文件转换为Markdown格式文本，包含OCR识别和公式提取。"""
    converter = _build_converter()
    result = converter.convert(pdf_path)
    return result.document.export_to_markdown()
