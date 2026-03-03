import { Crown, TrendingUp } from 'lucide-react';

interface UsageIndicatorProps {
  used: number;
  limit: number;
  isPremium: boolean;
  onUpgrade: () => void;
}

export default function UsageIndicator({ used, limit, isPremium, onUpgrade }: UsageIndicatorProps) {
  const percentage = isPremium ? 0 : (used / limit) * 100;
  const remaining = limit - used;
  const isNearLimit = remaining <= 2 && !isPremium;

  return (
    <div className={`bg-white rounded-2xl shadow-lg border p-6 mb-8 ${
      isNearLimit ? 'border-orange-300' : 'border-gray-200'
    }`}>
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
            {isPremium ? (
              <>
                <Crown className="w-5 h-5 text-yellow-500" />
                Premium Plan
              </>
            ) : (
              <>
                <TrendingUp className="w-5 h-5 text-purple-600" />
                Memory Usage
              </>
            )}
          </h3>
          <p className="text-sm text-gray-600 mt-1">
            {isPremium ? (
              'You have unlimited memories'
            ) : (
              <>
                {used} of {limit} free memories used
                {remaining > 0 && ` • ${remaining} remaining`}
              </>
            )}
          </p>
        </div>

        {!isPremium && (
          <button
            onClick={onUpgrade}
            className="px-4 py-2 bg-gradient-to-r from-purple-600 to-blue-600 text-white rounded-lg hover:shadow-lg transition-all text-sm font-medium flex items-center gap-2"
          >
            <Crown className="w-4 h-4" />
            Upgrade
          </button>
        )}
      </div>

      {!isPremium && (
        <>
          {/* Progress Bar */}
          <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden mb-2">
            <div
              className={`h-full transition-all duration-500 ${
                isNearLimit
                  ? 'bg-gradient-to-r from-orange-500 to-red-500'
                  : 'bg-gradient-to-r from-purple-600 to-blue-600'
              }`}
              style={{ width: `${Math.min(percentage, 100)}%` }}
            ></div>
          </div>

          {/* Warning Message */}
          {isNearLimit && (
            <div className="mt-4 p-3 bg-orange-50 border border-orange-200 rounded-lg">
              <p className="text-sm text-orange-800">
                ⚠️ You're running low on free memories. Upgrade to Premium for unlimited storage!
              </p>
            </div>
          )}

          {used >= limit && (
            <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg">
              <p className="text-sm text-red-800 font-medium">
                🚫 You've reached your free memory limit. Please upgrade to continue.
              </p>
            </div>
          )}
        </>
      )}

      {isPremium && (
        <div className="mt-4 p-3 bg-gradient-to-r from-purple-50 to-blue-50 border border-purple-200 rounded-lg">
          <p className="text-sm text-purple-800">
            ✨ Enjoying unlimited memories and premium features!
          </p>
        </div>
      )}
    </div>
  );
}
