import os
from pathlib import Path
from typing import List
import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader


class DocumentProcessor:
    def __init__(self, chunk_size: int = 600, chunk_overlap = 60):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len,
                                                            separators=['\n\n','\n'," ",""])


    def _load_excel(self, file_path: Path)->List[Document]:
        """
            Parses Excel spreadsheets row by row to preserve tabular data integrity.
            Each row is translated into a structured text string with its metadata.
        """
        documents = []
        try:
            # Read all sheets in the Excel file
            excel_file = pd.ExcelFile(file_path, engine='openpyxl')

            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name, dtype={"WorkBench Error Code": str})

                # Forward-fill missing values if necessary, or replace NaN with empty strings
                df = df.fillna("")
                for index, row in df.iterrows():
                    # Construct a clear, descriptive text string for the vector database
                    row_items = [f"{col}: {val} " for col, val in row.items() if str(val).strip()]
                    row_text = f"Sheet: {sheet_name} | Row {index + 1} Data:\n" + "\n".join(row_items)
                   #print(row_text)

                    # Store data structure as a LangChain document
                    doc = Document(
                        page_content=row_text,
                        metadata = {
                            "source": file_path.name,
                            "sheet": sheet_name,
                            "row": index + 1,
                            "error_code": str(row.get("WorkBench Error Code", "")).strip()
                        }
                    )

                    documents.append(doc)

        except Exception as e:
            print(f"Failed to parse Excel sheet {file_path.name}: {str(e)}")

        return documents


    def load_file(self, file_path: Path):
        """Loads a single document dynamically based on its extension."""
        ext = file_path.suffix.lower()
        try:
            if ext == ".pdf":
                loader = PyPDFLoader(str(file_path))
                return loader.load()
            elif ext in [".txt", ".md"]:
                loader = TextLoader(str(file_path), encoding="UTF-8")
                return loader.load()
            elif ext in ['.xlsx', ".xls"]:
                return self._load_excel(file_path)
            else:
                print(f"Skipping unsupported file type {file_path.name}")
                return []
        except Exception as e:
            print(f"Error loading {file_path.name}: str{e}")
            return []


    def process_directory(self, directory_path: Path):

        if directory_path.is_file():
            return self.load_file(directory_path)

        """Finds, loads, and structures all documentation from the source directory."""
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        raw_document = []
        excel_document = []

        for file_path in directory_path.iterdir():
            if file_path.is_file():
                ext = file_path.suffix.lower()
                if ext in ['.xlsx', 'xls']:
                    excel_document.extend(self.load_file(file_path))
                else:
                    raw_document.extend(self.load_file(file_path))

        # Split unstructured text (PDFs/TXTs) using the splitter
        processed_chunks = []
        if raw_document:
            processed_chunks.extend(self.text_splitter.split_documents(raw_document))

        if excel_document:
            processed_chunks.extend(excel_document)


        print(f"System prepared {len(processed_chunks)} total context node from {directory_path.name}")

        return processed_chunks
