import { useState } from 'react';
import { Calendar, Tag, Image as ImageIcon, MoreVertical, Trash2, Edit } from 'lucide-react';
import { toast } from 'sonner';

interface MemoryCardProps {
  memory: {
    id: number;
    title: string;
    content: string;
    category: string;
    tags: string[];
    timestamp: string;
    image_path?: string;
  };
}

export default function MemoryCard({ memory }: MemoryCardProps) {
  const [showMenu, setShowMenu] = useState(false);

  const categoryColors: { [key: string]: string } = {
    'Daily Life': 'bg-blue-100 text-blue-800',
    'Work & Meetings': 'bg-purple-100 text-purple-800',
    'Learning & Growth': 'bg-green-100 text-green-800',
    'Health & Fitness': 'bg-red-100 text-red-800',
    'Money & Shopping': 'bg-yellow-100 text-yellow-800',
    'Entertainment': 'bg-pink-100 text-pink-800',
    'Ideas & Creativity': 'bg-indigo-100 text-indigo-800',
    'General': 'bg-gray-100 text-gray-800',
  };

  const handleDelete = () => {
    toast.success('Memory deleted');
    setShowMenu(false);
  };

  const handleEdit = () => {
    toast.info('Edit feature coming soon');
    setShowMenu(false);
  };

  const formatDate = (timestamp: string) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;

    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: date.getFullYear() !== now.getFullYear() ? 'numeric' : undefined,
    });
  };

  return (
    <div className="bg-white border border-gray-200 rounded-xl p-5 hover:shadow-md transition-all group">
      <div className="flex items-start justify-between gap-4 mb-3">
        <div className="flex-1 min-w-0">
          <h3 className="font-semibold text-gray-900 mb-1 line-clamp-1">{memory.title}</h3>
          <div className="flex items-center gap-2 text-sm text-gray-600">
            <Calendar className="w-4 h-4" />
            <span>{formatDate(memory.timestamp)}</span>
          </div>
        </div>

        {/* Menu Button */}
        <div className="relative">
          <button
            onClick={() => setShowMenu(!showMenu)}
            className="p-1 text-gray-400 hover:text-gray-600 opacity-0 group-hover:opacity-100 transition-all"
          >
            <MoreVertical className="w-5 h-5" />
          </button>

          {/* Dropdown Menu */}
          {showMenu && (
            <>
              <div
                className="fixed inset-0 z-10"
                onClick={() => setShowMenu(false)}
              ></div>
              <div className="absolute right-0 top-8 z-20 w-40 bg-white border border-gray-200 rounded-lg shadow-lg overflow-hidden">
                <button
                  onClick={handleEdit}
                  className="w-full px-4 py-2 text-left text-sm text-gray-700 hover:bg-gray-50 flex items-center gap-2"
                >
                  <Edit className="w-4 h-4" />
                  Edit
                </button>
                <button
                  onClick={handleDelete}
                  className="w-full px-4 py-2 text-left text-sm text-red-600 hover:bg-red-50 flex items-center gap-2"
                >
                  <Trash2 className="w-4 h-4" />
                  Delete
                </button>
              </div>
            </>
          )}
        </div>
      </div>

      {/* Image Preview */}
      {memory.image_path && (
        <div className="mb-3 rounded-lg overflow-hidden">
          <img
            src={memory.image_path}
            alt={memory.title}
            className="w-full h-48 object-cover"
          />
        </div>
      )}

      {/* Content */}
      <p className="text-gray-700 mb-4 line-clamp-3">{memory.content}</p>

      {/* Footer */}
      <div className="flex items-center justify-between gap-3 flex-wrap">
        {/* Category Badge */}
        <span
          className={`px-3 py-1 rounded-full text-xs font-medium ${
            categoryColors[memory.category] || categoryColors['General']
          }`}
        >
          {memory.category}
        </span>

        {/* Tags */}
        {memory.tags.length > 0 && (
          <div className="flex items-center gap-2">
            <Tag className="w-4 h-4 text-gray-400" />
            <div className="flex gap-1 flex-wrap">
              {memory.tags.slice(0, 3).map((tag, index) => (
                <span
                  key={index}
                  className="px-2 py-0.5 bg-gray-100 text-gray-600 rounded text-xs"
                >
                  {tag}
                </span>
              ))}
              {memory.tags.length > 3 && (
                <span className="px-2 py-0.5 bg-gray-100 text-gray-600 rounded text-xs">
                  +{memory.tags.length - 3}
                </span>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
