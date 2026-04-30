import { useCallback, useRef, useState, type DragEvent, type KeyboardEvent } from "react";

interface Props {
  onUpload: (file: File) => void;
  disabled: boolean;
}

export function UploadZone({ onUpload, disabled }: Props) {
  const [dragOver, setDragOver] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback((file: File) => {
    setError(null);
    const lower = file.name.toLowerCase();
    if (!/\.(pdf|docx|pptx|xlsx)$/.test(lower)) {
      setError("Please select a PDF, Word (.docx), PowerPoint (.pptx), or Excel (.xlsx) file.");
      return;
    }
    setSelectedFile(file);
  }, []);

  const handleDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  };

  const handleKeyDown = (e: KeyboardEvent) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      inputRef.current?.click();
    }
  };

  return (
    <div className="flex flex-col items-center gap-6 w-full">
      {/* Drop zone */}
      <div
        role="button"
        tabIndex={0}
        aria-label="Upload a document. Drag and drop or press Enter to browse."
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        onKeyDown={handleKeyDown}
        className={`
          group relative flex w-full cursor-pointer flex-col items-center justify-center
          rounded-2xl border-2 border-dashed px-8 py-16 transition-all duration-300
          focus:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2 focus-visible:ring-offset-canvas
          ${dragOver
            ? "border-primary bg-primary/5 scale-[1.02]"
            : "border-elevated hover:border-primary/50 hover:bg-raised/50"
          }
          ${disabled ? "pointer-events-none opacity-50" : ""}
        `}
      >
        <div className={`mb-4 rounded-xl p-4 transition-colors ${dragOver ? "bg-primary/10" : "bg-raised"}`} aria-hidden="true">
          <svg className="h-8 w-8 text-primary" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
          </svg>
        </div>
        <p className="text-sm font-medium text-text">
          {dragOver ? "Drop your document here" : "Drag & drop a document here"}
        </p>
        <p className="mt-1 text-xs text-text-muted">
          or click to browse &mdash; PDF, Word, PowerPoint, or Excel
        </p>
        <input
          ref={inputRef}
          type="file"
          accept=".pdf,application/pdf,.docx,.pptx,.xlsx,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/vnd.openxmlformats-officedocument.presentationml.presentation,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
          className="sr-only"
          aria-label="Choose document file"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) handleFile(file);
          }}
        />
      </div>

      {/* Error */}
      {error && (
        <div role="alert" className="rounded-lg border border-failing/30 bg-failing/10 px-4 py-2 text-sm text-failing">
          {error}
        </div>
      )}

      {/* Selected file */}
      {selectedFile && (
        <div className="flex items-center gap-4 rounded-xl border border-elevated bg-raised px-5 py-3 animate-fade-up w-full">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10 flex-shrink-0" aria-hidden="true">
            <svg className="h-5 w-5 text-primary" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m2.25 0H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
            </svg>
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium text-text truncate">{selectedFile.name}</p>
            <p className="text-xs text-text-muted">
              {(selectedFile.size / 1024).toFixed(0)} KB
            </p>
          </div>
          <button
            onClick={() => onUpload(selectedFile)}
            disabled={disabled}
            className="rounded-xl bg-primary px-6 py-2.5 text-sm font-semibold text-white shadow-lg shadow-primary/20 hover:bg-primary-light focus:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2 focus-visible:ring-offset-canvas disabled:opacity-50 transition-all"
          >
            Remediate
          </button>
        </div>
      )}
    </div>
  );
}
