import React, { useState, useRef, useEffect } from 'react';
import { 
  MessageCircle, 
  Send, 
  Bot, 
  User, 
  Loader2,
  Lightbulb,
  ArrowRight,
  X
} from 'lucide-react';
import { useApi } from '../contexts/ApiContext';
import toast from 'react-hot-toast';

interface FollowUpQuestionsProps {
  initialQuery: string;
  onNewResearch: (query: string, results: any) => void;
  isSearching: boolean;
}

interface ConversationMessage {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  isFollowUp?: boolean;
}

const FollowUpQuestions: React.FC<FollowUpQuestionsProps> = ({ 
  initialQuery, 
  onNewResearch, 
  isSearching 
}) => {
  const [conversation, setConversation] = useState<ConversationMessage[]>([
    {
      id: '1',
      type: 'user',
      content: initialQuery,
      timestamp: new Date(),
      isFollowUp: false
    }
  ]);
  const [currentQuestion, setCurrentQuestion] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const { conductFollowUpResearch } = useApi();
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const suggestedQuestions = [
    "Can you provide more details about this?",
    "What are the latest developments in this area?",
    "How does this compare to similar topics?",
    "What are the potential implications?",
    "Are there any controversies or debates?",
    "What are the key challenges?",
    "Can you explain this in simpler terms?",
    "What are the future trends?"
  ];

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [conversation]);

  const handleFollowUp = async (question: string) => {
    if (!question.trim() || isProcessing || isSearching) return;

    const userMessage: ConversationMessage = {
      id: Date.now().toString(),
      type: 'user',
      content: question,
      timestamp: new Date(),
      isFollowUp: true
    };

    setConversation(prev => [...prev, userMessage]);
    setCurrentQuestion('');
    setIsProcessing(true);

    try {
      // Use the dedicated follow-up research endpoint
      const results = await conductFollowUpResearch(initialQuery, question);
      
      const assistantMessage: ConversationMessage = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: results.research_report?.summary || 'I found some additional information about your question.',
        timestamp: new Date(),
        isFollowUp: true
      };

      setConversation(prev => [...prev, assistantMessage]);
      
      // Trigger new research display
      onNewResearch(question, results);
      
      toast.success('Follow-up research completed');
    } catch (error) {
      console.error('Follow-up research failed:', error);
      toast.error('Failed to process follow-up question');
      
      const errorMessage: ConversationMessage = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: 'Sorry, I encountered an error while processing your follow-up question. Please try again.',
        timestamp: new Date(),
        isFollowUp: true
      };
      
      setConversation(prev => [...prev, errorMessage]);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleSuggestionClick = (suggestion: string) => {
    setCurrentQuestion(suggestion);
    setShowSuggestions(false);
  };

  const clearConversation = () => {
    setConversation([
      {
        id: '1',
        type: 'user',
        content: initialQuery,
        timestamp: new Date(),
        isFollowUp: false
      }
    ]);
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-gray-700 dark:to-gray-600 px-6 py-4 border-b border-gray-200 dark:border-gray-600">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-indigo-100 dark:bg-indigo-900 rounded-lg">
              <MessageCircle className="w-6 h-6 text-indigo-600 dark:text-indigo-400" />
            </div>
            <div>
              <h3 className="text-xl font-bold text-gray-900 dark:text-white">
                Interactive Research
              </h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Ask follow-up questions to dig deeper
              </p>
            </div>
          </div>
          <button
            onClick={clearConversation}
            className="flex items-center space-x-2 px-3 py-1 text-sm text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
          >
            <X className="w-4 h-4" />
            <span>Clear</span>
          </button>
        </div>
      </div>

      {/* Conversation History */}
      <div className="max-h-96 overflow-y-auto p-6 space-y-4">
        {conversation.map((message) => (
          <div
            key={message.id}
            className={`flex items-start space-x-3 ${
              message.type === 'user' ? 'justify-end' : 'justify-start'
            }`}
          >
            {message.type === 'assistant' && (
              <div className="flex-shrink-0 w-8 h-8 bg-indigo-100 dark:bg-indigo-900 rounded-full flex items-center justify-center">
                <Bot className="w-4 h-4 text-indigo-600 dark:text-indigo-400" />
              </div>
            )}
            
            <div
              className={`max-w-xs lg:max-w-md px-4 py-3 rounded-lg ${
                message.type === 'user'
                  ? 'bg-indigo-500 text-white'
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-900 dark:text-white'
              }`}
            >
              <p className="text-sm leading-relaxed">{message.content}</p>
              <div className="flex items-center justify-between mt-2">
                <span className="text-xs opacity-70">
                  {message.timestamp.toLocaleTimeString()}
                </span>
                {message.isFollowUp && (
                  <span className="text-xs opacity-70 bg-white/20 dark:bg-gray-600/20 px-2 py-1 rounded">
                    Follow-up
                  </span>
                )}
              </div>
            </div>
            
            {message.type === 'user' && (
              <div className="flex-shrink-0 w-8 h-8 bg-gray-100 dark:bg-gray-700 rounded-full flex items-center justify-center">
                <User className="w-4 h-4 text-gray-600 dark:text-gray-400" />
              </div>
            )}
          </div>
        ))}
        
        {isProcessing && (
          <div className="flex items-start space-x-3">
            <div className="flex-shrink-0 w-8 h-8 bg-indigo-100 dark:bg-indigo-900 rounded-full flex items-center justify-center">
              <Bot className="w-4 h-4 text-indigo-600 dark:text-indigo-400" />
            </div>
            <div className="bg-gray-100 dark:bg-gray-700 px-4 py-3 rounded-lg">
              <div className="flex items-center space-x-2">
                <Loader2 className="w-4 h-4 animate-spin text-indigo-600" />
                <span className="text-sm text-gray-600 dark:text-gray-400">
                  Researching your question...
                </span>
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Input Section */}
      <div className="border-t border-gray-200 dark:border-gray-600 p-4">
        <div className="flex items-center space-x-3">
          <div className="flex-1 relative">
            <input
              type="text"
              value={currentQuestion}
              onChange={(e) => setCurrentQuestion(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleFollowUp(currentQuestion)}
              placeholder="Ask a follow-up question..."
              className="w-full px-4 py-3 pr-12 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400"
              disabled={isProcessing || isSearching}
            />
            <button
              onClick={() => setShowSuggestions(!showSuggestions)}
              className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
            >
              <Lightbulb className="w-5 h-5" />
            </button>
          </div>
          
          <button
            onClick={() => handleFollowUp(currentQuestion)}
            disabled={!currentQuestion.trim() || isProcessing || isSearching}
            className="px-4 py-3 bg-indigo-500 text-white rounded-lg hover:bg-indigo-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center space-x-2"
          >
            {isProcessing ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Send className="w-4 h-4" />
            )}
            <span>Ask</span>
          </button>
        </div>

        {/* Suggested Questions */}
        {showSuggestions && (
          <div className="mt-4 p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
            <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
              Suggested questions:
            </h4>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              {suggestedQuestions.map((suggestion, index) => (
                <button
                  key={index}
                  onClick={() => handleSuggestionClick(suggestion)}
                  className="text-left p-2 text-sm text-gray-600 dark:text-gray-400 hover:bg-white dark:hover:bg-gray-600 rounded transition-colors flex items-center space-x-2"
                >
                  <ArrowRight className="w-3 h-3 flex-shrink-0" />
                  <span>{suggestion}</span>
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default FollowUpQuestions;
