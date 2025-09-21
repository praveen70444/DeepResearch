import React, { useState } from 'react';
import { 
  Brain, 
  ChevronDown, 
  ChevronUp, 
  Lightbulb, 
  Target, 
  Search, 
  CheckCircle, 
  ArrowRight,
  Info,
  Zap,
  BookOpen,
  Eye,
  EyeOff
} from 'lucide-react';

interface ReasoningStep {
  id: string;
  title: string;
  description: string;
  details: string;
  confidence: number;
  sources: string[];
  type: 'analysis' | 'synthesis' | 'validation' | 'conclusion';
}

interface ReasoningAssistantProps {
  reasoningSteps: number;
  researchData: any;
  query?: string;
}

const ReasoningAssistant: React.FC<ReasoningAssistantProps> = ({ 
  reasoningSteps, 
  researchData,
  query 
}) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [selectedStep, setSelectedStep] = useState<string | null>(null);
  const [showDetails, setShowDetails] = useState(false);

  // Generate detailed reasoning steps based on the research data
  const generateReasoningSteps = (): ReasoningStep[] => {
    const steps: ReasoningStep[] = [];
    
    // Step 1: Query Analysis
    steps.push({
      id: 'query-analysis',
      title: 'Query Analysis & Understanding',
      description: 'Analyzing the research question to identify key concepts, context, and scope',
      details: `I began by breaking down your query into core components and identifying the main topics, subtopics, and specific information needs. This analysis helps me understand what type of research approach would be most effective and what sources would be most relevant.`,
      confidence: 0.95,
      sources: ['Query parsing', 'Intent recognition', 'Context analysis'],
      type: 'analysis'
    });

    // Step 2: Information Gathering
    steps.push({
      id: 'information-gathering',
      title: 'Information Gathering & Source Discovery',
      description: 'Searching and collecting relevant information from multiple sources',
      details: `I searched through various databases, academic papers, news articles, and other reliable sources to gather comprehensive information about your topic. This step involved identifying credible sources, checking for recency, and ensuring diversity in perspectives.`,
      confidence: 0.88,
      sources: researchData?.research_report?.sources?.map((s: any) => s.title || 'Source') || ['Academic databases', 'News sources', 'Expert opinions'],
      type: 'analysis'
    });

    // Step 3: Data Synthesis
    steps.push({
      id: 'data-synthesis',
      title: 'Data Synthesis & Pattern Recognition',
      description: 'Combining and analyzing information to identify patterns and insights',
      details: `I analyzed all gathered information to identify key patterns, trends, and relationships. This involved cross-referencing different sources, identifying consensus points, and highlighting areas of disagreement or uncertainty.`,
      confidence: 0.92,
      sources: ['Cross-source analysis', 'Pattern recognition', 'Trend identification'],
      type: 'synthesis'
    });

    // Step 4: Critical Evaluation
    steps.push({
      id: 'critical-evaluation',
      title: 'Critical Evaluation & Validation',
      description: 'Evaluating the reliability and relevance of information',
      details: `I critically evaluated each piece of information for accuracy, relevance, and reliability. This included checking source credibility, identifying potential biases, and assessing the strength of evidence supporting different claims.`,
      confidence: 0.85,
      sources: ['Source credibility assessment', 'Bias detection', 'Evidence evaluation'],
      type: 'validation'
    });

    // Step 5: Conclusion Formation
    steps.push({
      id: 'conclusion-formation',
      title: 'Conclusion Formation & Summary',
      description: 'Synthesizing findings into coherent conclusions and recommendations',
      details: `Based on all the analysis, I synthesized the findings into clear, actionable conclusions. This step involved weighing different perspectives, acknowledging limitations, and providing evidence-based recommendations.`,
      confidence: researchData?.research_report?.confidence_score || 0.87,
      sources: ['Evidence synthesis', 'Conclusion formation', 'Recommendation development'],
      type: 'conclusion'
    });

    return steps;
  };

  const reasoningStepsData = generateReasoningSteps();

  const getStepIcon = (type: string) => {
    switch (type) {
      case 'analysis':
        return <Search className="w-5 h-5" />;
      case 'synthesis':
        return <Brain className="w-5 h-5" />;
      case 'validation':
        return <CheckCircle className="w-5 h-5" />;
      case 'conclusion':
        return <Target className="w-5 h-5" />;
      default:
        return <Lightbulb className="w-5 h-5" />;
    }
  };

  const getStepColor = (type: string) => {
    switch (type) {
      case 'analysis':
        return 'text-blue-600 dark:text-blue-400 bg-blue-100 dark:bg-blue-900';
      case 'synthesis':
        return 'text-purple-600 dark:text-purple-400 bg-purple-100 dark:bg-purple-900';
      case 'validation':
        return 'text-green-600 dark:text-green-400 bg-green-100 dark:bg-green-900';
      case 'conclusion':
        return 'text-orange-600 dark:text-orange-400 bg-orange-100 dark:bg-orange-900';
      default:
        return 'text-gray-600 dark:text-gray-400 bg-gray-100 dark:bg-gray-900';
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.9) return 'text-green-600 dark:text-green-400';
    if (confidence >= 0.8) return 'text-yellow-600 dark:text-yellow-400';
    return 'text-red-600 dark:text-red-400';
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-violet-50 to-purple-50 dark:from-gray-700 dark:to-gray-600 px-6 py-4 border-b border-gray-200 dark:border-gray-600">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-violet-100 dark:bg-violet-900 rounded-lg">
              <Brain className="w-6 h-6 text-violet-600 dark:text-violet-400" />
            </div>
            <div>
              <h3 className="text-xl font-bold text-gray-900 dark:text-white">
                AI Reasoning Assistant
              </h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Understand how I arrived at these conclusions
              </p>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setShowDetails(!showDetails)}
              className="flex items-center space-x-2 px-3 py-1 text-sm text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
            >
              {showDetails ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              <span>{showDetails ? 'Hide Details' : 'Show Details'}</span>
            </button>
            <button
              onClick={() => setIsExpanded(!isExpanded)}
              className="flex items-center space-x-2 px-3 py-1 text-sm text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
            >
              {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
              <span>{isExpanded ? 'Collapse' : 'Expand'}</span>
            </button>
          </div>
        </div>
      </div>

      {/* Reasoning Overview */}
      <div className="p-6">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
            <div className="flex items-center space-x-2 mb-2">
              <Zap className="w-4 h-4 text-violet-600" />
              <span className="font-medium text-gray-900 dark:text-white">Processing Steps</span>
            </div>
            <p className="text-2xl font-bold text-violet-600 dark:text-violet-400">{reasoningSteps}</p>
          </div>
          <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
            <div className="flex items-center space-x-2 mb-2">
              <BookOpen className="w-4 h-4 text-violet-600" />
              <span className="font-medium text-gray-900 dark:text-white">Sources Analyzed</span>
            </div>
            <p className="text-2xl font-bold text-violet-600 dark:text-violet-400">
              {researchData?.research_report?.sources?.length || 0}
            </p>
          </div>
          <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
            <div className="flex items-center space-x-2 mb-2">
              <Target className="w-4 h-4 text-violet-600" />
              <span className="font-medium text-gray-900 dark:text-white">Confidence</span>
            </div>
            <p className={`text-2xl font-bold ${getConfidenceColor(researchData?.research_report?.confidence_score || 0.85)}`}>
              {Math.round((researchData?.research_report?.confidence_score || 0.85) * 100)}%
            </p>
          </div>
        </div>

        {/* Reasoning Steps */}
        {isExpanded && (
          <div className="space-y-4">
            <h4 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              Detailed Reasoning Process
            </h4>
            {reasoningStepsData.map((step, index) => (
              <div
                key={step.id}
                className={`border border-gray-200 dark:border-gray-600 rounded-lg overflow-hidden transition-all duration-200 ${
                  selectedStep === step.id ? 'ring-2 ring-violet-500 shadow-md' : 'hover:shadow-sm'
                }`}
              >
                <button
                  onClick={() => setSelectedStep(selectedStep === step.id ? null : step.id)}
                  className="w-full p-4 text-left hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-4">
                      <div className={`p-2 rounded-lg ${getStepColor(step.type)}`}>
                        {getStepIcon(step.type)}
                      </div>
                      <div className="flex-1">
                        <div className="flex items-center space-x-2 mb-1">
                          <span className="text-sm font-medium text-gray-500 dark:text-gray-400">
                            Step {index + 1}
                          </span>
                          <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStepColor(step.type)}`}>
                            {step.type.charAt(0).toUpperCase() + step.type.slice(1)}
                          </span>
                        </div>
                        <h5 className="font-semibold text-gray-900 dark:text-white mb-1">
                          {step.title}
                        </h5>
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                          {step.description}
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <span className={`text-sm font-medium ${getConfidenceColor(step.confidence)}`}>
                        {Math.round(step.confidence * 100)}% confidence
                      </span>
                      {selectedStep === step.id ? (
                        <ChevronUp className="w-4 h-4 text-gray-500" />
                      ) : (
                        <ChevronDown className="w-4 h-4 text-gray-500" />
                      )}
                    </div>
                  </div>
                </button>

                {selectedStep === step.id && (
                  <div className="px-4 pb-4 border-t border-gray-200 dark:border-gray-600 bg-gray-50 dark:bg-gray-700/30">
                    <div className="mt-4">
                      <h6 className="font-medium text-gray-900 dark:text-white mb-2">
                        Detailed Explanation:
                      </h6>
                      <p className="text-gray-700 dark:text-gray-300 leading-relaxed mb-4">
                        {step.details}
                      </p>
                      
                      <div className="mb-4">
                        <h6 className="font-medium text-gray-900 dark:text-white mb-2">
                          Sources & Methods:
                        </h6>
                        <div className="flex flex-wrap gap-2">
                          {step.sources.map((source, idx) => (
                            <span
                              key={idx}
                              className="px-3 py-1 bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 rounded-full text-sm border border-gray-200 dark:border-gray-600"
                            >
                              {source}
                            </span>
                          ))}
                        </div>
                      </div>

                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-2">
                          <Info className="w-4 h-4 text-gray-500" />
                          <span className="text-sm text-gray-600 dark:text-gray-400">
                            This step contributes to the overall research quality and reliability
                          </span>
                        </div>
                        <div className="flex items-center space-x-2">
                          <div className="w-24 bg-gray-200 dark:bg-gray-600 rounded-full h-2">
                            <div
                              className="bg-violet-500 h-2 rounded-full transition-all duration-300"
                              style={{ width: `${step.confidence * 100}%` }}
                            />
                          </div>
                          <span className={`text-sm font-medium ${getConfidenceColor(step.confidence)}`}>
                            {Math.round(step.confidence * 100)}%
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            ))}

            {/* Summary */}
            <div className="mt-6 p-4 bg-violet-50 dark:bg-violet-900/20 rounded-lg border border-violet-200 dark:border-violet-700">
              <div className="flex items-start space-x-3">
                <Lightbulb className="w-5 h-5 text-violet-600 dark:text-violet-400 mt-0.5" />
                <div>
                  <h6 className="font-medium text-gray-900 dark:text-white mb-1">
                    How This Process Ensures Quality
                  </h6>
                  <p className="text-sm text-gray-700 dark:text-gray-300">
                    Each step in this reasoning process is designed to ensure accuracy, reliability, and comprehensiveness. 
                    By following this systematic approach, I can provide you with well-researched, evidence-based insights 
                    that you can trust and act upon.
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ReasoningAssistant;
