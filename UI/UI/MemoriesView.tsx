import { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router';
import { Brain, Search, Filter, Calendar, ArrowLeft, Sparkles } from 'lucide-react';
import MemoryCard from './MemoryCard';
import { toast } from 'sonner';

export default function MemoriesView() {
  const navigate = useNavigate();
  const [user, setUser] = useState<any>(null);
  const [memories, setMemories] = useState<any[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [sortBy, setSortBy] = useState<'newest' | 'oldest'>('newest');
  const [loading, setLoading] = useState(false);

  const categories = [
    'All',
    'Daily Life',
    'Work & Meetings',
    'Learning & Growth',
    'Health & Fitness',
    'Money & Shopping',
    'Entertainment',
    'Ideas & Creativity',
    'General',
  ];

  // Mock memories data
  const mockMemories = [
    {
      id: 1,
      title: 'Team meeting about Q1 goals',
      content: 'Discussed Q1 objectives with the team. Need to focus on user growth and product improvements. Key action items: increase conversion rate by 15%, launch two new features, and improve customer retention.',
      category: 'Work & Meetings',
      tags: ['meeting', 'goals', 'team', 'Q1'],
      timestamp: '2026-03-01T10:30:00',
      image_path: '',
    },
    {
      id: 2,
      title: 'Morning workout routine ideas',
      content: 'Tried a new HIIT workout today. 20 minutes of cardio followed by strength training. Feeling great! Should make this a daily habit.',
      category: 'Health & Fitness',
      tags: ['workout', 'fitness', 'HIIT', 'exercise'],
      timestamp: '2026-03-01T07:00:00',
    },
    {
      id: 3,
      title: 'New feature idea for app',
      content: 'What if we added voice search to make finding memories even easier? Could use speech recognition API. Also thinking about collaborative memories feature.',
      category: 'Ideas & Creativity',
      tags: ['idea', 'feature', 'voice search', 'innovation'],
      timestamp: '2026-02-28T15:45:00',
    },
    {
      id: 4,
      title: 'Book notes - Atomic Habits',
      content: 'Chapter 3: Small changes can have remarkable results. Focus on systems, not goals. The aggregation of marginal gains.',
      category: 'Learning & Growth',
      tags: ['book', 'notes', 'habits', 'self-improvement'],
      timestamp: '2026-02-27T20:15:00',
    },
    {
      id: 5,
      title: 'Weekend trip planning',
      content: 'Planning a trip to the mountains. Need to book accommodation, pack hiking gear, and check weather forecast.',
      category: 'Daily Life',
      tags: ['travel', 'planning', 'weekend', 'mountains'],
      timestamp: '2026-02-26T18:00:00',
    },
    {
      id: 6,
      title: 'Monthly budget review',
      content: 'Reviewed February expenses. Saved 20% more than last month. Need to reduce dining out expenses. Investment portfolio performing well.',
      category: 'Money & Shopping',
      tags: ['budget', 'finance', 'savings', 'review'],
      timestamp: '2026-02-25T16:30:00',
    },
  ];

  useEffect(() => {
    const userStr = sessionStorage.getItem('user');
    if (!userStr) {
      navigate('/login');
      return;
    }
    setUser(JSON.parse(userStr));
    setMemories(mockMemories);
  }, [navigate]);

  const handleSearch = () => {
    if (!searchQuery.trim()) {
      toast.info('Please enter a search query');
      return;
    }

    setLoading(true);
    setTimeout(() => {
      setLoading(false);
      toast.success(`Found ${filteredMemories.length} results`);
    }, 500);
  };

  const handleGenerateSummary = () => {
    toast.success('AI Summary: This week you focused on work goals, fitness, and learning. You had 3 productive meetings and completed 2 book chapters.');
  };

  const filteredMemories = memories
    .filter((memory) => {
      const matchesSearch =
        searchQuery === '' ||
        memory.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
        memory.content.toLowerCase().includes(searchQuery.toLowerCase()) ||
        memory.tags.some((tag: string) => tag.toLowerCase().includes(searchQuery.toLowerCase()));

      const matchesCategory =
        selectedCategory === 'all' ||
        memory.category.toLowerCase() === selectedCategory.toLowerCase();

      return matchesSearch && matchesCategory;
    })
    .sort((a, b) => {
      const dateA = new Date(a.timestamp).getTime();
      const dateB = new Date(b.timestamp).getTime();
      return sortBy === 'newest' ? dateB - dateA : dateA - dateB;
    });

  if (!user) {
    return null;
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-white to-blue-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <Link to="/dashboard" className="flex items-center gap-2 text-gray-600 hover:text-purple-600">
              <ArrowLeft className="w-5 h-5" />
              <span>Back to Dashboard</span>
            </Link>

            <Link to="/" className="flex items-center gap-2">
              <Brain className="w-8 h-8 text-purple-600" />
              <span className="text-xl font-bold bg-gradient-to-r from-purple-600 to-blue-600 bg-clip-text text-transparent">
                MemoAI
              </span>
            </Link>

            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-gradient-to-br from-purple-600 to-blue-600 rounded-full flex items-center justify-center text-white font-semibold">
                {user.name.charAt(0)}
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Page Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">All Memories</h1>
          <p className="text-gray-600">
            {memories.length} {memories.length === 1 ? 'memory' : 'memories'} stored
          </p>
        </div>

        {/* Search and Filters */}
        <div className="bg-white rounded-2xl shadow-lg border border-gray-200 p-6 mb-8">
          {/* Search Bar */}
          <div className="flex gap-3 mb-6">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
              <input
                type="text"
                placeholder="Search memories by content, tags, or keywords..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
                className="w-full pl-10 pr-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
              />
            </div>
            <button
              onClick={handleSearch}
              disabled={loading}
              className="px-6 py-3 bg-gradient-to-r from-purple-600 to-blue-600 text-white rounded-lg hover:shadow-lg transition-all disabled:opacity-50 font-medium"
            >
              {loading ? 'Searching...' : 'Search'}
            </button>
          </div>

          {/* Filters */}
          <div className="flex flex-col sm:flex-row gap-4">
            {/* Category Filter */}
            <div className="flex-1">
              <label className="block text-sm font-medium text-gray-700 mb-2">Category</label>
              <div className="relative">
                <Filter className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
                <select
                  value={selectedCategory}
                  onChange={(e) => setSelectedCategory(e.target.value)}
                  className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 bg-white"
                >
                  {categories.map((category) => (
                    <option key={category} value={category.toLowerCase()}>
                      {category}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {/* Sort Filter */}
            <div className="flex-1">
              <label className="block text-sm font-medium text-gray-700 mb-2">Sort By</label>
              <div className="relative">
                <Calendar className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
                <select
                  value={sortBy}
                  onChange={(e) => setSortBy(e.target.value as 'newest' | 'oldest')}
                  className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 bg-white"
                >
                  <option value="newest">Newest First</option>
                  <option value="oldest">Oldest First</option>
                </select>
              </div>
            </div>
          </div>

          {/* AI Summary Button */}
          <div className="mt-6 pt-6 border-t border-gray-200">
            <button
              onClick={handleGenerateSummary}
              className="w-full sm:w-auto px-6 py-3 bg-gradient-to-r from-purple-100 to-blue-100 text-purple-700 rounded-lg hover:from-purple-200 hover:to-blue-200 transition-all font-medium flex items-center justify-center gap-2"
            >
              <Sparkles className="w-5 h-5" />
              Generate AI Summary
            </button>
          </div>
        </div>

        {/* Results Count */}
        {searchQuery && (
          <div className="mb-4">
            <p className="text-gray-600">
              Found <span className="font-semibold text-purple-600">{filteredMemories.length}</span> result
              {filteredMemories.length !== 1 ? 's' : ''}
              {searchQuery && ` for "${searchQuery}"`}
            </p>
          </div>
        )}

        {/* Memory Grid */}
        {filteredMemories.length > 0 ? (
          <div className="grid gap-4">
            {filteredMemories.map((memory) => (
              <MemoryCard key={memory.id} memory={memory} />
            ))}
          </div>
        ) : (
          <div className="bg-white rounded-2xl shadow-lg border border-gray-200 p-12 text-center">
            <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <Search className="w-8 h-8 text-gray-400" />
            </div>
            <h3 className="text-lg font-medium text-gray-900 mb-2">No memories found</h3>
            <p className="text-gray-600 mb-6">
              {searchQuery
                ? 'Try adjusting your search terms or filters'
                : 'Start capturing your thoughts to see them here'}
            </p>
            <Link
              to="/dashboard"
              className="inline-block px-6 py-3 bg-gradient-to-r from-purple-600 to-blue-600 text-white rounded-lg hover:shadow-lg transition-all font-medium"
            >
              Create Your First Memory
            </Link>
          </div>
        )}
      </main>
    </div>
  );
}
