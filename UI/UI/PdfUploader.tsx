import { useState, useRef } from 'react';
import { FileText, Upload, X, CheckCircle } from 'lucide-react';
import { toast } from 'sonner';

interface PdfUploaderProps {
  onMemorySaved: (memory: any) => void;
}

export default function PdfUploader({ onMemorySaved }: PdfUploaderProps) {
  const [pdfFile, setPdfFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingProgress, setProcessingProgress] = useState(0);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    if (file.type !== 'application/pdf') {
      toast.error('Please select a PDF file');
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      toast.error('PDF size must be less than 10MB');
      return;
    }

    setPdfFile(file);
  };

  const removeFile = () => {
    setPdfFile(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const processPdf = async () => {
    if (!pdfFile) {
      toast.error('Please select a PDF file first');
      return;
    }

    setIsProcessing(true);
    setProcessingProgress(0);

    try {
      // Simulate PDF processing with progress
      for (let i = 0; i <= 100; i += 10) {
        await new Promise(resolve => setTimeout(resolve, 200));
        setProcessingProgress(i);
      }

      // Create mock memory
      const newMemory = {
        id: Date.now(),
        title: pdfFile.name.replace('.pdf', ''),
        content: `PDF document "${pdfFile.name}" has been processed and indexed for search.`,
        category: 'Learning & Growth',
        tags: ['pdf', 'document', 'learning'],
        timestamp: new Date().toISOString(),
        image_path: '',
      };

      onMemorySaved(newMemory);
      removeFile();
      setProcessingProgress(0);
      toast.success('PDF processed successfully! All pages are now searchable.');
    } catch (error) {
      toast.error('Failed to process PDF');
      setProcessingProgress(0);
    } finally {
      setIsProcessing(false);
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(2) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
  };

  return (
    <div className="space-y-6">
      {/* Upload Area */}
      {!pdfFile && (
        <div>
          <button
            onClick={() => fileInputRef.current?.click()}
            className="w-full p-12 border-2 border-dashed border-gray-300 rounded-lg hover:border-purple-500 hover:bg-purple-50 transition-all group"
          >
            <div className="flex flex-col items-center gap-4">
              <div className="w-20 h-20 bg-purple-100 rounded-full flex items-center justify-center group-hover:scale-110 transition-transform">
                <FileText className="w-10 h-10 text-purple-600" />
              </div>
              <div>
                <p className="text-lg font-medium text-gray-900 mb-1">Upload PDF Document</p>
                <p className="text-sm text-gray-600">Click to browse or drag and drop</p>
                <p className="text-xs text-gray-500 mt-2">Maximum file size: 10MB</p>
              </div>
            </div>
          </button>

          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf"
            onChange={handleFileSelect}
            className="hidden"
          />
        </div>
      )}

      {/* File Info */}
      {pdfFile && !isProcessing && (
        <div className="space-y-4">
          <div className="flex items-start gap-4 p-4 bg-gray-50 border border-gray-300 rounded-lg">
            <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center flex-shrink-0">
              <FileText className="w-6 h-6 text-purple-600" />
            </div>
            <div className="flex-1 min-w-0">
              <p className="font-medium text-gray-900 truncate">{pdfFile.name}</p>
              <p className="text-sm text-gray-600">{formatFileSize(pdfFile.size)}</p>
            </div>
            <button
              onClick={removeFile}
              className="p-2 text-gray-400 hover:text-red-500 transition-colors flex-shrink-0"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
            <h4 className="text-sm font-medium text-blue-900 mb-2">What happens next:</h4>
            <ul className="space-y-1 text-sm text-blue-800">
              <li className="flex items-center gap-2">
                <CheckCircle className="w-4 h-4 flex-shrink-0" />
                Text will be extracted from all pages
              </li>
              <li className="flex items-center gap-2">
                <CheckCircle className="w-4 h-4 flex-shrink-0" />
                Content will be chunked and indexed
              </li>
              <li className="flex items-center gap-2">
                <CheckCircle className="w-4 h-4 flex-shrink-0" />
                You can search through the entire document
              </li>
            </ul>
          </div>

          <button
            onClick={processPdf}
            className="w-full px-6 py-3 bg-gradient-to-r from-purple-600 to-blue-600 text-white rounded-lg hover:shadow-lg transition-all font-medium"
          >
            Process PDF
          </button>
        </div>
      )}

      {/* Processing State */}
      {isProcessing && (
        <div className="space-y-4">
          <div className="flex items-center gap-4 p-4 bg-gray-50 border border-gray-300 rounded-lg">
            <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center flex-shrink-0">
              <FileText className="w-6 h-6 text-purple-600" />
            </div>
            <div className="flex-1">
              <p className="font-medium text-gray-900 mb-2">Processing {pdfFile?.name}</p>
              
              {/* Progress Bar */}
              <div className="w-full bg-gray-200 rounded-full h-2 overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-purple-600 to-blue-600 transition-all duration-300"
                  style={{ width: `${processingProgress}%` }}
                ></div>
              </div>
              
              <p className="text-sm text-gray-600 mt-2">{processingProgress}% complete</p>
            </div>
          </div>

          <div className="p-4 bg-purple-50 border border-purple-200 rounded-lg">
            <div className="flex items-center gap-3">
              <div className="w-5 h-5 border-2 border-purple-600 border-t-transparent rounded-full animate-spin"></div>
              <div>
                <p className="text-sm font-medium text-purple-900">Processing your document...</p>
                <p className="text-xs text-purple-700">This may take a few moments</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Info Box */}
      {!pdfFile && (
        <div className="p-4 bg-gray-50 border border-gray-200 rounded-lg">
          <h4 className="text-sm font-medium text-gray-900 mb-2">💡 Supported Features:</h4>
          <ul className="space-y-1 text-sm text-gray-600">
            <li>• Extract text from multi-page PDFs</li>
            <li>• Smart chunking for better search results</li>
            <li>• Full-text semantic search across all pages</li>
            <li>• Automatic categorization and tagging</li>
          </ul>
        </div>
      )}
    </div>
  );
}
