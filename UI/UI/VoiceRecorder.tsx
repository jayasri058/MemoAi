import { useState, useRef, useEffect } from 'react';
import { Mic, Square, Play, Pause } from 'lucide-react';
import { toast } from 'sonner';

interface VoiceRecorderProps {
  onMemorySaved: (memory: any) => void;
}

export default function VoiceRecorder({ onMemorySaved }: VoiceRecorderProps) {
  const [isRecording, setIsRecording] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [duration, setDuration] = useState(0);
  const recognitionRef = useRef<any>(null);
  const timerRef = useRef<any>(null);

  useEffect(() => {
    // Check if browser supports speech recognition
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    
    if (SpeechRecognition) {
      recognitionRef.current = new SpeechRecognition();
      recognitionRef.current.continuous = true;
      recognitionRef.current.interimResults = true;
      recognitionRef.current.lang = 'en-US';

      recognitionRef.current.onresult = (event: any) => {
        let interimTranscript = '';
        let finalTranscript = '';

        for (let i = event.resultIndex; i < event.results.length; i++) {
          const transcript = event.results[i][0].transcript;
          if (event.results[i].isFinal) {
            finalTranscript += transcript + ' ';
          } else {
            interimTranscript += transcript;
          }
        }

        setTranscript((prev) => {
          const newTranscript = prev + finalTranscript;
          return newTranscript;
        });
      };

      recognitionRef.current.onerror = (event: any) => {
        console.error('Speech recognition error:', event.error);
        if (event.error === 'no-speech') {
          toast.info('No speech detected. Please try again.');
        } else {
          toast.error('Speech recognition error. Please try again.');
        }
        stopRecording();
      };
    }

    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    };
  }, []);

  const startRecording = () => {
    if (!recognitionRef.current) {
      toast.error('Speech recognition not supported in this browser');
      return;
    }

    setIsRecording(true);
    setIsPaused(false);
    setTranscript('');
    setDuration(0);

    try {
      recognitionRef.current.start();
      
      timerRef.current = setInterval(() => {
        setDuration((prev) => prev + 1);
      }, 1000);
    } catch (error) {
      toast.error('Failed to start recording');
      setIsRecording(false);
    }
  };

  const pauseRecording = () => {
    if (recognitionRef.current) {
      recognitionRef.current.stop();
      setIsPaused(true);
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    }
  };

  const resumeRecording = () => {
    if (recognitionRef.current) {
      recognitionRef.current.start();
      setIsPaused(false);
      
      timerRef.current = setInterval(() => {
        setDuration((prev) => prev + 1);
      }, 1000);
    }
  };

  const stopRecording = () => {
    if (recognitionRef.current) {
      recognitionRef.current.stop();
    }
    
    if (timerRef.current) {
      clearInterval(timerRef.current);
    }

    setIsRecording(false);
    setIsPaused(false);
  };

  const saveMemory = async () => {
    if (!transcript.trim()) {
      toast.error('Please record some content first');
      return;
    }

    // Create mock memory
    const newMemory = {
      id: Date.now(),
      title: transcript.split(' ').slice(0, 5).join(' ') + '...',
      content: transcript,
      category: 'General',
      tags: ['voice', 'recording'],
      timestamp: new Date().toISOString(),
      image_path: '',
    };

    onMemorySaved(newMemory);
    setTranscript('');
    setDuration(0);
  };

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="space-y-6">
      {/* Recording Controls */}
      <div className="flex flex-col items-center gap-6">
        {/* Duration Display */}
        {isRecording && (
          <div className="text-3xl font-bold text-purple-600 tabular-nums">
            {formatDuration(duration)}
          </div>
        )}

        {/* Microphone Button */}
        <div className="relative">
          {isRecording && (
            <div className="absolute inset-0 rounded-full bg-red-500 animate-ping opacity-75"></div>
          )}
          <button
            onClick={isRecording ? stopRecording : startRecording}
            className={`relative w-24 h-24 rounded-full flex items-center justify-center transition-all shadow-lg ${
              isRecording
                ? 'bg-red-500 hover:bg-red-600'
                : 'bg-gradient-to-br from-purple-600 to-blue-600 hover:shadow-xl'
            }`}
          >
            {isRecording ? (
              <Square className="w-10 h-10 text-white" />
            ) : (
              <Mic className="w-10 h-10 text-white" />
            )}
          </button>
        </div>

        {/* Pause/Resume Button */}
        {isRecording && (
          <button
            onClick={isPaused ? resumeRecording : pauseRecording}
            className="px-6 py-2 bg-gray-200 hover:bg-gray-300 rounded-lg flex items-center gap-2 transition-colors"
          >
            {isPaused ? (
              <>
                <Play className="w-4 h-4" />
                Resume
              </>
            ) : (
              <>
                <Pause className="w-4 h-4" />
                Pause
              </>
            )}
          </button>
        )}

        {/* Status */}
        <p className="text-sm text-gray-600">
          {isRecording
            ? isPaused
              ? 'Recording paused - Click Resume to continue'
              : 'Recording... Click the square to stop'
            : 'Click the microphone to start recording'}
        </p>
      </div>

      {/* Transcript Display */}
      {transcript && (
        <div className="border border-gray-300 rounded-lg p-4 bg-gray-50">
          <h3 className="text-sm font-medium text-gray-700 mb-2">Transcript:</h3>
          <p className="text-gray-900 whitespace-pre-wrap">{transcript}</p>
        </div>
      )}

      {/* Save Button */}
      {transcript && !isRecording && (
        <button
          onClick={saveMemory}
          className="w-full px-6 py-3 bg-gradient-to-r from-purple-600 to-blue-600 text-white rounded-lg hover:shadow-lg transition-all font-medium"
        >
          Save Memory
        </button>
      )}

      {/* Browser Compatibility Note */}
      {!('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) && (
        <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
          <p className="text-sm text-yellow-800">
            ⚠️ Speech recognition is not supported in this browser. Please use Chrome, Edge, or Safari.
          </p>
        </div>
      )}
    </div>
  );
}
