import { X, Crown, Check, Zap } from 'lucide-react';
import { toast } from 'sonner';

interface PremiumModalProps {
  onClose: () => void;
}

export default function PremiumModal({ onClose }: PremiumModalProps) {
  const handleUpgrade = () => {
    toast.success('Upgrade successful! You now have unlimited memories.');
    onClose();
  };

  const features = [
    'Unlimited memories',
    'AI-powered memory summaries',
    'Advanced search filters',
    'Priority customer support',
    'Export your memories',
    'Custom categories',
    'Ad-free experience',
    'Early access to new features',
  ];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm">
      <div className="bg-white rounded-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto shadow-2xl">
        {/* Header */}
        <div className="sticky top-0 bg-gradient-to-r from-purple-600 to-blue-600 text-white p-6 rounded-t-2xl">
          <div className="flex items-start justify-between">
            <div>
              <div className="flex items-center gap-2 mb-2">
                <Crown className="w-8 h-8" />
                <h2 className="text-2xl font-bold">Upgrade to Premium</h2>
              </div>
              <p className="text-purple-100">Unlock unlimited memories and premium features</p>
            </div>
            <button
              onClick={onClose}
              className="p-2 hover:bg-white/20 rounded-lg transition-colors"
            >
              <X className="w-6 h-6" />
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="p-6">
          {/* Pricing */}
          <div className="text-center mb-8">
            <div className="inline-block p-1 bg-gradient-to-r from-purple-600 to-blue-600 rounded-2xl mb-4">
              <div className="bg-white rounded-xl px-8 py-6">
                <div className="text-5xl font-bold text-gray-900 mb-2">
                  ₹299
                  <span className="text-xl text-gray-600 font-normal">/month</span>
                </div>
                <p className="text-sm text-gray-600">or $4.99/month</p>
              </div>
            </div>
          </div>

          {/* Features Grid */}
          <div className="mb-8">
            <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
              <Zap className="w-5 h-5 text-purple-600" />
              What's Included
            </h3>
            <div className="grid sm:grid-cols-2 gap-3">
              {features.map((feature, index) => (
                <div key={index} className="flex items-start gap-2">
                  <Check className="w-5 h-5 text-green-500 flex-shrink-0 mt-0.5" />
                  <span className="text-gray-700">{feature}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Comparison */}
          <div className="bg-gray-50 rounded-xl p-6 mb-8">
            <h3 className="font-semibold text-gray-900 mb-4">Free vs Premium</h3>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-sm font-medium text-gray-600 mb-2">Free</p>
                <ul className="space-y-1 text-sm text-gray-700">
                  <li>• 10 memories</li>
                  <li>• Basic search</li>
                  <li>• Standard support</li>
                </ul>
              </div>
              <div>
                <p className="text-sm font-medium text-purple-600 mb-2">Premium</p>
                <ul className="space-y-1 text-sm text-gray-700">
                  <li>• Unlimited memories</li>
                  <li>• Advanced AI search</li>
                  <li>• Priority support</li>
                </ul>
              </div>
            </div>
          </div>

          {/* CTA Buttons */}
          <div className="space-y-3">
            <button
              onClick={handleUpgrade}
              className="w-full px-6 py-4 bg-gradient-to-r from-purple-600 to-blue-600 text-white rounded-xl hover:shadow-xl transition-all font-semibold text-lg flex items-center justify-center gap-2"
            >
              <Crown className="w-5 h-5" />
              Upgrade Now
            </button>
            <button
              onClick={onClose}
              className="w-full px-6 py-3 border-2 border-gray-300 text-gray-700 rounded-xl hover:bg-gray-50 transition-all font-medium"
            >
              Maybe Later
            </button>
          </div>

          {/* Security Note */}
          <p className="text-xs text-gray-500 text-center mt-6">
            🔒 Secure payment • Cancel anytime • 30-day money-back guarantee
          </p>
        </div>
      </div>
    </div>
  );
}
