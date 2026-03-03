import { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router';
import {
  Brain,
  Mic,
  Camera,
  FileText,
  Search,
  Plus,
  LogOut,
  Crown,
  Settings,
  Menu,
  X,
  Sparkles,
} from 'lucide-react';
import { toast } from 'sonner';
import VoiceRecorder from './VoiceRecorder';
import ImageUploader from './ImageUploader';
import PdfUploader from './PdfUploader';
import MemoryCard from './MemoryCard';
import UsageIndicator from './UsageIndicator';
import PremiumModal from './PremiumModal';

export default function Dashboard() {
  const navigate = useNavigate();
  const [user, setUser] = useState<any>(null);
  const [activeTab, setActiveTab] = useState<'voice' | 'image' | 'pdf'>('voice');
  const [memories, setMemories] = useState<any[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [showPremiumModal, setShowPremiumModal] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [usageStats, setUsageStats] = useState({
    used: 3,
    limit: 10,
    isPremium: false,
  });

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
      content: 'Discussed Q1 objectives with the team. Need to focus on user growth and product improvements.',
      category: 'Work & Meetings',
      tags: ['meeting', 'goals', 'team', 'Q1'],
      timestamp: '2026-03-01T10:30:00',
      image_path: '',
    },
    {
      id: 2,
      title: 'Morning workout routine ideas',
      content: 'Tried a new HIIT workout today. 20 minutes of cardio followed by strength training.',
      category: 'Health & Fitness',
      tags: ['workout', 'fitness', 'HIIT', 'exercise'],
      timestamp: '2026-03-01T07:00:00',
    },
    {
      id: 3,
      title: 'New feature idea for app',
      content: 'What if we added voice search to make finding memories even easier? Could use speech recognition API.',
      category: 'Ideas & Creativity',
      tags: ['idea', 'feature', 'voice search', 'innovation'],
      timestamp: '2026-02-28T15:45:00',
    },
  ];

  useEffect(() => {
    // Check if user is logged in
    const userStr = sessionStorage.getItem('user');
    if (!userStr) {
      navigate('/login');
      return;
    }
    setUser(JSON.parse(userStr));
    setMemories(mockMemories);
  }, [navigate]);

  const handleLogout = () => {
    sessionStorage.removeItem('user');
    toast.success('Logged out successfully');
    navigate('/');
  };

  const handleMemorySaved = (memory: any) => {
    setMemories([memory, ...memories]);
    setUsageStats(prev => ({ ...prev, used: prev.used + 1 }));
    toast.success('Memory saved successfully!');
  };

  const filteredMemories = memories.filter((memory) => {
    const matchesSearch =
      searchQuery === '' ||
      memory.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      memory.content.toLowerCase().includes(searchQuery.toLowerCase()) ||
      memory.tags.some((tag: string) => tag.toLowerCase().includes(searchQuery.toLowerCase()));

    const matchesCategory =
      selectedCategory === 'all' ||
      memory.category.toLowerCase() === selectedCategory.toLowerCase();

    return matchesSearch && matchesCategory;
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
            {/* Logo */}
            <Link to="/" className="flex items-center gap-2">
              <Brain className="w-8 h-8 text-purple-600" />
              <span className="text-xl font-bold bg-gradient-to-r from-purple-600 to-blue-600 bg-clip-text text-transparent">
                MemoAI
              </span>
            </Link>

            {/* Desktop Navigation */}
            <nav className="hidden md:flex items-center gap-6">
              <Link
                to="/dashboard"
                className="text-purple-600 font-medium border-b-2 border-purple-600 pb-1"
              >
                Dashboard
              </Link>
              <Link to="/memories" className="text-gray-600 hover:text-purple-600 transition-colors">
                All Memories
              </Link>
              <Link to="/contact" className="text-gray-600 hover:text-purple-600 transition-colors">
                Contact
              </Link>
            </nav>

            {/* User Menu */}
            <div className="flex items-center gap-4">
              {!usageStats.isPremium && (
                <button
                  onClick={() => setShowPremiumModal(true)}
                  className="hidden md:flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-purple-600 to-blue-600 text-white rounded-lg hover:shadow-lg transition-all"
                >
                  <Crown className="w-4 h-4" />
                  Upgrade
                </button>
              )}

              <div className="hidden md:flex items-center gap-3">
                <div className="w-10 h-10 bg-gradient-to-br from-purple-600 to-blue-600 rounded-full flex items-center justify-center text-white font-semibold">
                  {user.name.charAt(0)}
                </div>
                <div className="hidden lg:block">
                  <div className="text-sm font-medium text-gray-900">{user.name}</div>
                  <div className="text-xs text-gray-500">{user.email}</div>
                </div>
              </div>

              <button
                onClick={handleLogout}
                className="hidden md:flex items-center gap-2 px-4 py-2 text-gray-600 hover:text-purple-600 transition-colors"
              >
                <LogOut className="w-4 h-4" />
                Logout
              </button>

              {/* Mobile menu button */}
              <button
                onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                className="md:hidden p-2 text-gray-600"
              >
                {mobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
              </button>
            </div>
          </div>
        </div>

        {/* Mobile Menu */}
        {mobileMenuOpen && (
          <div className="md:hidden border-t border-gray-200 bg-white">
            <div className="px-4 py-4 space-y-4">
              <Link
                to="/dashboard"
                className="block text-purple-600 font-medium"
                onClick={() => setMobileMenuOpen(false)}
              >
                Dashboard
              </Link>
              <Link
                to="/memories"
                className="block text-gray-600"
                onClick={() => setMobileMenuOpen(false)}
              >
                All Memories
              </Link>
              <Link
                to="/contact"
                className="block text-gray-600"
                onClick={() => setMobileMenuOpen(false)}
              >
                Contact
              </Link>
              <button
                onClick={() => {
                  setShowPremiumModal(true);
                  setMobileMenuOpen(false);
                }}
                className="flex items-center gap-2 text-purple-600"
              >
                <Crown className="w-4 h-4" />
                Upgrade to Premium
              </button>
              <button onClick={handleLogout} className="flex items-center gap-2 text-gray-600">
                <LogOut className="w-4 h-4" />
                Logout
              </button>
            </div>
          </div>
        )}
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Welcome Section */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            Welcome back, {user.name.split(' ')[0]}! 👋
          </h1>
          <p className="text-gray-600">Capture your thoughts, ideas, and moments.</p>
        </div>

        {/* Usage Indicator */}
        <UsageIndicator
          used={usageStats.used}
          limit={usageStats.limit}
          isPremium={usageStats.isPremium}
          onUpgrade={() => setShowPremiumModal(true)}
        />

        {/* Capture Section */}
        <div className="bg-white rounded-2xl shadow-lg border border-gray-200 p-6 mb-8">
          <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
            <Plus className="w-6 h-6 text-purple-600" />
            Create New Memory
          </h2>

          {/* Tabs */}
          <div className="flex gap-2 mb-6 border-b border-gray-200">
            <button
              onClick={() => setActiveTab('voice')}
              className={`flex items-center gap-2 px-4 py-3 border-b-2 transition-colors ${
                activeTab === 'voice'
                  ? 'border-purple-600 text-purple-600'
                  : 'border-transparent text-gray-600 hover:text-purple-600'
              }`}
            >
              <Mic className="w-5 h-5" />
              Voice
            </button>
            <button
              onClick={() => setActiveTab('image')}
              className={`flex items-center gap-2 px-4 py-3 border-b-2 transition-colors ${
                activeTab === 'image'
                  ? 'border-purple-600 text-purple-600'
                  : 'border-transparent text-gray-600 hover:text-purple-600'
              }`}
            >
              <Camera className="w-5 h-5" />
              Image
            </button>
            <button
              onClick={() => setActiveTab('pdf')}
              className={`flex items-center gap-2 px-4 py-3 border-b-2 transition-colors ${
                activeTab === 'pdf'
                  ? 'border-purple-600 text-purple-600'
                  : 'border-transparent text-gray-600 hover:text-purple-600'
              }`}
            >
              <FileText className="w-5 h-5" />
              PDF
            </button>
          </div>

          {/* Tab Content */}
          {activeTab === 'voice' && <VoiceRecorder onMemorySaved={handleMemorySaved} />}
          {activeTab === 'image' && <ImageUploader onMemorySaved={handleMemorySaved} />}
          {activeTab === 'pdf' && <PdfUploader onMemorySaved={handleMemorySaved} />}
        </div>

        {/* Memories Section */}
        <div className="bg-white rounded-2xl shadow-lg border border-gray-200 p-6">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-6">
            <h2 className="text-xl font-bold flex items-center gap-2">
              <Sparkles className="w-6 h-6 text-purple-600" />
              Recent Memories
            </h2>

            {/* Search */}
            <div className="relative flex-1 max-w-md">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
              <input
                type="text"
                placeholder="Search memories..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
              />
            </div>
          </div>

          {/* Category Filter */}
          <div className="flex gap-2 mb-6 overflow-x-auto pb-2 scrollbar-thin">
            {categories.map((category) => (
              <button
                key={category}
                onClick={() => setSelectedCategory(category.toLowerCase())}
                className={`px-4 py-2 rounded-lg whitespace-nowrap transition-colors ${
                  selectedCategory === category.toLowerCase()
                    ? 'bg-purple-600 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                {category}
              </button>
            ))}
          </div>

          {/* Memory Cards */}
          {filteredMemories.length > 0 ? (
            <div className="grid gap-4">
              {filteredMemories.map((memory) => (
                <MemoryCard key={memory.id} memory={memory} />
              ))}
            </div>
          ) : (
            <div className="text-center py-12">
              <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <Search className="w-8 h-8 text-gray-400" />
              </div>
              <h3 className="text-lg font-medium text-gray-900 mb-2">No memories found</h3>
              <p className="text-gray-600">
                {searchQuery
                  ? 'Try a different search term'
                  : 'Start capturing your thoughts to see them here'}
              </p>
            </div>
          )}

          {filteredMemories.length > 0 && (
            <div className="mt-6 text-center">
              <Link
                to="/memories"
                className="inline-flex items-center gap-2 text-purple-600 hover:text-purple-700 font-medium"
              >
                View All Memories
                <Search className="w-4 h-4" />
              </Link>
            </div>
          )}
        </div>
      </main>

      {/* Premium Modal */}
      {showPremiumModal && <PremiumModal onClose={() => setShowPremiumModal(false)} />}
    </div>
  );
}
