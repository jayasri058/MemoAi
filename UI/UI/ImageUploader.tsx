import { useState, useRef } from 'react';
import { Camera, Upload, X, Image as ImageIcon } from 'lucide-react';
import { toast } from 'sonner';

interface ImageUploaderProps {
  onMemorySaved: (memory: any) => void;
}

export default function ImageUploader({ onMemorySaved }: ImageUploaderProps) {
  const [imagePreview, setImagePreview] = useState<string>('');
  const [caption, setCaption] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const cameraInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    if (!file.type.startsWith('image/')) {
      toast.error('Please select an image file');
      return;
    }

    if (file.size > 5 * 1024 * 1024) {
      toast.error('Image size must be less than 5MB');
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      setImagePreview(e.target?.result as string);
    };
    reader.readAsDataURL(file);
  };

  const removeImage = () => {
    setImagePreview('');
    setCaption('');
    if (fileInputRef.current) fileInputRef.current.value = '';
    if (cameraInputRef.current) cameraInputRef.current.value = '';
  };

  const saveMemory = async () => {
    if (!imagePreview) {
      toast.error('Please select an image first');
      return;
    }

    setIsProcessing(true);

    try {
      // Simulate AI processing
      await new Promise(resolve => setTimeout(resolve, 2000));

      // Create mock memory with AI-generated description
      const aiDescription = 'A captured moment showing interesting visual content.';
      const content = caption || aiDescription;

      const newMemory = {
        id: Date.now(),
        title: content.split(' ').slice(0, 5).join(' ') + '...',
        content: content,
        category: 'General',
        tags: ['image', 'visual', 'photo'],
        timestamp: new Date().toISOString(),
        image_path: imagePreview,
      };

      onMemorySaved(newMemory);
      removeImage();
      toast.success('Memory saved with AI analysis!');
    } catch (error) {
      toast.error('Failed to process image');
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Upload Buttons */}
      {!imagePreview && (
        <div className="grid md:grid-cols-2 gap-4">
          {/* File Upload */}
          <button
            onClick={() => fileInputRef.current?.click()}
            className="p-8 border-2 border-dashed border-gray-300 rounded-lg hover:border-purple-500 hover:bg-purple-50 transition-all group"
          >
            <div className="flex flex-col items-center gap-3">
              <div className="w-16 h-16 bg-purple-100 rounded-full flex items-center justify-center group-hover:scale-110 transition-transform">
                <Upload className="w-8 h-8 text-purple-600" />
              </div>
              <div>
                <p className="font-medium text-gray-900">Upload from Device</p>
                <p className="text-sm text-gray-600">PNG, JPG up to 5MB</p>
              </div>
            </div>
          </button>

          {/* Camera Capture */}
          <button
            onClick={() => cameraInputRef.current?.click()}
            className="p-8 border-2 border-dashed border-gray-300 rounded-lg hover:border-purple-500 hover:bg-purple-50 transition-all group"
          >
            <div className="flex flex-col items-center gap-3">
              <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center group-hover:scale-110 transition-transform">
                <Camera className="w-8 h-8 text-blue-600" />
              </div>
              <div>
                <p className="font-medium text-gray-900">Take Photo</p>
                <p className="text-sm text-gray-600">Use your camera</p>
              </div>
            </div>
          </button>

          {/* Hidden File Inputs */}
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileSelect}
            className="hidden"
          />
          <input
            ref={cameraInputRef}
            type="file"
            accept="image/*"
            capture="environment"
            onChange={handleFileSelect}
            className="hidden"
          />
        </div>
      )}

      {/* Image Preview */}
      {imagePreview && (
        <div className="space-y-4">
          <div className="relative rounded-lg overflow-hidden border border-gray-300">
            <img
              src={imagePreview}
              alt="Preview"
              className="w-full h-auto max-h-96 object-contain bg-gray-50"
            />
            <button
              onClick={removeImage}
              className="absolute top-2 right-2 p-2 bg-red-500 text-white rounded-full hover:bg-red-600 transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          {/* Caption Input */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Add Caption (Optional)
            </label>
            <textarea
              value={caption}
              onChange={(e) => setCaption(e.target.value)}
              placeholder="Describe this image or add context..."
              rows={3}
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 resize-none"
            />
          </div>

          {/* Save Button */}
          <button
            onClick={saveMemory}
            disabled={isProcessing}
            className="w-full px-6 py-3 bg-gradient-to-r from-purple-600 to-blue-600 text-white rounded-lg hover:shadow-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed font-medium flex items-center justify-center gap-2"
          >
            {isProcessing ? (
              <>
                <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                Processing with AI...
              </>
            ) : (
              <>
                <ImageIcon className="w-5 h-5" />
                Save Memory
              </>
            )}
          </button>

          {isProcessing && (
            <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
              <p className="text-sm text-blue-800 text-center">
                🤖 AI is analyzing your image...
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
