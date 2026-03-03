import { Link } from 'react-router';
import { Brain, Mic, Camera, FileText, Search, Sparkles, Check, ArrowRight } from 'lucide-react';
import { ImageWithFallback } from './figma/ImageWithFallback';

export default function Landing() {
  const features = [
    {
      icon: Mic,
      title: 'Voice Capture',
      description: 'Speak your thoughts naturally and let AI transcribe and organize them instantly.',
    },
    {
      icon: Camera,
      title: 'Image Analysis',
      description: 'Upload images and get AI-powered descriptions and context automatically.',
    },
    {
      icon: FileText,
      title: 'PDF Processing',
      description: 'Upload PDFs and extract text with smart chunking for easy searching.',
    },
    {
      icon: Search,
      title: 'Semantic Search',
      description: 'Find memories using natural language with vector-powered similarity search.',
    },
    {
      icon: Sparkles,
      title: 'AI Categorization',
      description: 'Automatic classification into 8 smart categories with AI-generated tags.',
    },
    {
      icon: Brain,
      title: 'Memory Summaries',
      description: 'Get AI-generated summaries of your recent thoughts and activities.',
    },
  ];

  const categories = [
    'Daily Life',
    'Work & Meetings',
    'Learning & Growth',
    'Health & Fitness',
    'Money & Shopping',
    'Entertainment',
    'Ideas & Creativity',
    'General',
  ];

  return (
    <div className="min-h-screen bg-gradient-to-b from-purple-50 to-white">
      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 bg-white/80 backdrop-blur-lg border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center gap-2">
              <Brain className="w-8 h-8 text-purple-600" />
              <span className="text-xl font-bold bg-gradient-to-r from-purple-600 to-blue-600 bg-clip-text text-transparent">
                MemoAI
              </span>
            </div>
            <div className="flex items-center gap-4">
              <Link
                to="/login"
                className="px-4 py-2 text-gray-700 hover:text-purple-600 transition-colors"
              >
                Login
              </Link>
              <Link
                to="/register"
                className="px-6 py-2 bg-gradient-to-r from-purple-600 to-blue-600 text-white rounded-lg hover:shadow-lg transition-all"
              >
                Get Started
              </Link>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="pt-32 pb-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            <div>
              <div className="inline-block px-4 py-1 bg-purple-100 text-purple-700 rounded-full text-sm font-medium mb-6">
                Your AI-Powered Second Brain
              </div>
              <h1 className="text-5xl lg:text-6xl font-bold mb-6 leading-tight">
                Never Forget
                <br />
                <span className="bg-gradient-to-r from-purple-600 to-blue-600 bg-clip-text text-transparent">
                  Anything Again
                </span>
              </h1>
              <p className="text-xl text-gray-600 mb-8 leading-relaxed">
                MemoAI intelligently captures, categorizes, and retrieves your thoughts, conversations,
                and visual memories using advanced AI and vector search technology.
              </p>
              <div className="flex flex-col sm:flex-row gap-4">
                <Link
                  to="/register"
                  className="px-8 py-4 bg-gradient-to-r from-purple-600 to-blue-600 text-white rounded-lg hover:shadow-xl transition-all flex items-center justify-center gap-2 group"
                >
                  Start Free Trial
                  <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                </Link>
                <Link
                  to="/dashboard"
                  className="px-8 py-4 border-2 border-purple-600 text-purple-600 rounded-lg hover:bg-purple-50 transition-all flex items-center justify-center gap-2"
                >
                  View Demo
                </Link>
              </div>
              <div className="mt-8 flex items-center gap-6 text-sm text-gray-600">
                <div className="flex items-center gap-2">
                  <Check className="w-5 h-5 text-green-500" />
                  <span>10 Free Memories</span>
                </div>
                <div className="flex items-center gap-2">
                  <Check className="w-5 h-5 text-green-500" />
                  <span>No Credit Card</span>
                </div>
              </div>
            </div>
            <div className="relative">
              <div className="absolute inset-0 bg-gradient-to-r from-purple-400 to-blue-400 rounded-3xl blur-3xl opacity-20"></div>
              <ImageWithFallback
                src="https://images.unsplash.com/photo-1737505599159-5ffc1dcbc08f?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxicmFpbiUyMG1lbW9yeSUyMGFydGlmaWNpYWwlMjBpbnRlbGxpZ2VuY2UlMjBkaWdpdGFsfGVufDF8fHx8MTc3MjM0NzM1NHww&ixlib=rb-4.1.0&q=80&w=1080"
                alt="AI Brain Memory"
                className="relative rounded-3xl shadow-2xl w-full"
              />
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold mb-4">Powerful Features</h2>
            <p className="text-xl text-gray-600">Everything you need to manage your memories</p>
          </div>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {features.map((feature, index) => (
              <div
                key={index}
                className="p-6 rounded-2xl border border-gray-200 hover:border-purple-300 hover:shadow-lg transition-all group"
              >
                <div className="w-12 h-12 bg-gradient-to-br from-purple-100 to-blue-100 rounded-xl flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                  <feature.icon className="w-6 h-6 text-purple-600" />
                </div>
                <h3 className="text-xl font-semibold mb-2">{feature.title}</h3>
                <p className="text-gray-600">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Categories Section */}
      <section className="py-20 bg-gradient-to-b from-white to-purple-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold mb-4">Smart Categorization</h2>
            <p className="text-xl text-gray-600">AI automatically organizes your memories into 8 categories</p>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {categories.map((category, index) => (
              <div
                key={index}
                className="px-6 py-4 bg-white rounded-xl shadow-sm hover:shadow-md transition-all text-center font-medium text-gray-700 border border-gray-100"
              >
                {category}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold mb-4">How It Works</h2>
            <p className="text-xl text-gray-600">Simple, fast, and intelligent</p>
          </div>
          <div className="grid md:grid-cols-3 gap-12">
            <div className="text-center">
              <div className="w-16 h-16 bg-gradient-to-br from-purple-600 to-blue-600 text-white rounded-full flex items-center justify-center text-2xl font-bold mx-auto mb-6">
                1
              </div>
              <h3 className="text-xl font-semibold mb-3">Capture</h3>
              <p className="text-gray-600">
                Speak, upload images, or add PDFs. MemoAI captures everything instantly.
              </p>
            </div>
            <div className="text-center">
              <div className="w-16 h-16 bg-gradient-to-br from-purple-600 to-blue-600 text-white rounded-full flex items-center justify-center text-2xl font-bold mx-auto mb-6">
                2
              </div>
              <h3 className="text-xl font-semibold mb-3">Process</h3>
              <p className="text-gray-600">
                AI analyzes, categorizes, and generates tags automatically with advanced ML.
              </p>
            </div>
            <div className="text-center">
              <div className="w-16 h-16 bg-gradient-to-br from-purple-600 to-blue-600 text-white rounded-full flex items-center justify-center text-2xl font-bold mx-auto mb-6">
                3
              </div>
              <h3 className="text-xl font-semibold mb-3">Retrieve</h3>
              <p className="text-gray-600">
                Search using natural language and find exactly what you need, instantly.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Pricing */}
      <section className="py-20 bg-gradient-to-b from-purple-50 to-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold mb-4">Simple Pricing</h2>
            <p className="text-xl text-gray-600">Start free, upgrade when you need more</p>
          </div>
          <div className="grid md:grid-cols-2 gap-8 max-w-4xl mx-auto">
            {/* Free Plan */}
            <div className="p-8 rounded-2xl border-2 border-gray-200 bg-white">
              <h3 className="text-2xl font-bold mb-2">Free</h3>
              <div className="text-4xl font-bold mb-6">
                $0<span className="text-lg text-gray-600 font-normal">/month</span>
              </div>
              <ul className="space-y-3 mb-8">
                <li className="flex items-center gap-2">
                  <Check className="w-5 h-5 text-green-500 flex-shrink-0" />
                  <span>10 memories</span>
                </li>
                <li className="flex items-center gap-2">
                  <Check className="w-5 h-5 text-green-500 flex-shrink-0" />
                  <span>Voice, image & PDF capture</span>
                </li>
                <li className="flex items-center gap-2">
                  <Check className="w-5 h-5 text-green-500 flex-shrink-0" />
                  <span>AI categorization</span>
                </li>
                <li className="flex items-center gap-2">
                  <Check className="w-5 h-5 text-green-500 flex-shrink-0" />
                  <span>Semantic search</span>
                </li>
              </ul>
              <Link
                to="/register"
                className="block w-full px-6 py-3 border-2 border-purple-600 text-purple-600 rounded-lg hover:bg-purple-50 transition-all text-center font-medium"
              >
                Get Started
              </Link>
            </div>

            {/* Premium Plan */}
            <div className="p-8 rounded-2xl border-2 border-purple-600 bg-gradient-to-br from-purple-50 to-blue-50 relative">
              <div className="absolute -top-4 right-8 px-4 py-1 bg-gradient-to-r from-purple-600 to-blue-600 text-white rounded-full text-sm font-medium">
                Popular
              </div>
              <h3 className="text-2xl font-bold mb-2">Premium</h3>
              <div className="text-4xl font-bold mb-6">
                ₹299<span className="text-lg text-gray-600 font-normal">/month</span>
              </div>
              <ul className="space-y-3 mb-8">
                <li className="flex items-center gap-2">
                  <Check className="w-5 h-5 text-green-500 flex-shrink-0" />
                  <span className="font-semibold">Unlimited memories</span>
                </li>
                <li className="flex items-center gap-2">
                  <Check className="w-5 h-5 text-green-500 flex-shrink-0" />
                  <span>Everything in Free</span>
                </li>
                <li className="flex items-center gap-2">
                  <Check className="w-5 h-5 text-green-500 flex-shrink-0" />
                  <span>AI memory summaries</span>
                </li>
                <li className="flex items-center gap-2">
                  <Check className="w-5 h-5 text-green-500 flex-shrink-0" />
                  <span>Priority support</span>
                </li>
              </ul>
              <Link
                to="/register"
                className="block w-full px-6 py-3 bg-gradient-to-r from-purple-600 to-blue-600 text-white rounded-lg hover:shadow-lg transition-all text-center font-medium"
              >
                Upgrade to Premium
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-900 text-white py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid md:grid-cols-4 gap-8 mb-8">
            <div>
              <div className="flex items-center gap-2 mb-4">
                <Brain className="w-8 h-8 text-purple-400" />
                <span className="text-xl font-bold">MemoAI</span>
              </div>
              <p className="text-gray-400">Your AI-powered second brain for capturing and retrieving memories.</p>
            </div>
            <div>
              <h4 className="font-semibold mb-4">Product</h4>
              <ul className="space-y-2 text-gray-400">
                <li><Link to="/dashboard" className="hover:text-white">Features</Link></li>
                <li><Link to="/register" className="hover:text-white">Pricing</Link></li>
                <li><Link to="/dashboard" className="hover:text-white">Demo</Link></li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold mb-4">Company</h4>
              <ul className="space-y-2 text-gray-400">
                <li><Link to="/contact" className="hover:text-white">Contact</Link></li>
                <li><a href="#" className="hover:text-white">About</a></li>
                <li><a href="#" className="hover:text-white">Privacy</a></li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold mb-4">Support</h4>
              <ul className="space-y-2 text-gray-400">
                <li><a href="#" className="hover:text-white">Documentation</a></li>
                <li><a href="#" className="hover:text-white">API</a></li>
                <li><Link to="/contact" className="hover:text-white">Help Center</Link></li>
              </ul>
            </div>
          </div>
          <div className="border-t border-gray-800 pt-8 text-center text-gray-400">
            <p>&copy; 2026 MemoAI. All rights reserved. Built with AI technology.</p>
          </div>
        </div>
      </footer>
    </div>
  );
}
