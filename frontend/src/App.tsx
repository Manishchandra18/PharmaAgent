// frontend/src/App.tsx
import { useState } from "react";
import { FileUpload } from "./components/FileUpload";
import { SummaryGenerator } from "./components/SummaryGenerator";

// Function to format extracted text into readable paragraphs and bullet points
const formatExtractedText = (text: string) => {
  if (!text) return null;
  
  // Split text into lines and clean up
  const lines = text.split('\n').map(line => line.trim()).filter(line => line.length > 0);
  
  // Group lines into logical sections
  const sections: string[] = [];
  let currentSection: string[] = [];
  
  for (const line of lines) {
    // Check if line looks like a header/title (all caps, short, or ends with colon)
    const isHeader = line.length < 50 && (
      line === line.toUpperCase() || 
      line.endsWith(':') || 
      line.match(/^[A-Z][a-z\s]+:$/) ||
      line.match(/^(Patient|Doctor|Date|Time|Medication|Dosage|Frequency|Age|Gender|Diagnosis|Treatment|Prescription)/i)
    );
    
    if (isHeader && currentSection.length > 0) {
      // Save current section and start new one
      sections.push(currentSection.join(' '));
      currentSection = [line];
    } else {
      currentSection.push(line);
    }
  }
  
  // Add the last section
  if (currentSection.length > 0) {
    sections.push(currentSection.join(' '));
  }
  
  // Format sections as paragraphs or bullet points
  return sections.map((section, index) => {
    const trimmedSection = section.trim();
    
    // Check if section looks like a list (contains multiple items separated by common delimiters)
    const listIndicators = ['•', '-', '*', '1.', '2.', '3.', 'a.', 'b.', 'c.'];
    const hasListIndicators = listIndicators.some(indicator => trimmedSection.includes(indicator));
    
    // Check if section contains medication/dosage patterns
    const hasMedicationPattern = /(mg|ml|tablet|capsule|dose|frequency|twice|daily|weekly)/i.test(trimmedSection);
    
    if (hasListIndicators || hasMedicationPattern) {
      // Format as bullet points
      const items = trimmedSection.split(/(?=[•\-*]|\d+\.|[a-z]\.)/).filter(item => item.trim().length > 0);
      return (
        <div key={index} className="mb-3">
          {items.map((item, itemIndex) => (
            <div key={itemIndex} className="flex items-start mb-1">
              <span className="text-blue-600 mr-2 mt-0.5">•</span>
              <span className="text-gray-700">{item.trim()}</span>
            </div>
          ))}
        </div>
      );
    } else {
      // Format as paragraph
      return (
        <p key={index} className="mb-3 text-gray-700 leading-relaxed">
          {trimmedSection}
        </p>
      );
    }
  });
};

type ChatRole = 'user' | 'assistant';

interface ChatMessage {
  role: ChatRole;
  content: string;
}

function App() {
  const [extractedText, setExtractedText] = useState<string | null>(null);
  const [qualityData, setQualityData] = useState<any>(null);
  const [medicalAnalysis, setMedicalAnalysis] = useState<any>(null);
  const [showTextPreview, setShowTextPreview] = useState<boolean>(false);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState<string>('');
  const [isChatLoading, setIsChatLoading] = useState<boolean>(false);
  const [chatError, setChatError] = useState<string | null>(null);

  const handleReset = () => {
    setExtractedText(null);
    setQualityData(null);
    setMedicalAnalysis(null);
    setShowTextPreview(false);
  };

  const handleSendChat = async (e: React.FormEvent) => {
    e.preventDefault();
    const trimmedInput = chatInput.trim();
    if (!trimmedInput) return;

    const newMessages: ChatMessage[] = [
      ...chatMessages,
      { role: 'user', content: trimmedInput },
    ];

    setChatMessages(newMessages);
    setChatInput('');
    setChatError(null);
    setIsChatLoading(true);

    try {
      const response = await fetch('http://localhost:8000/api/v1/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ messages: newMessages }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        throw new Error(errorData?.detail || 'Chat request failed');
      }

      const data = await response.json();
      if (!data.reply) {
        throw new Error('No reply received from assistant');
      }

      setChatMessages((prev) => [
        ...prev,
        { role: 'assistant', content: data.reply as string },
      ]);
    } catch (error) {
      if (error instanceof Error) {
        setChatError(error.message);
      } else {
        setChatError('An unknown error occurred while sending the message.');
      }
    } finally {
      setIsChatLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-red-50 via-white to-orange-50 flex flex-col">
      {/* Header */}
      <header className="bg-white shadow-lg border-b-4 border-red-600 flex-shrink-0">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center space-x-6">
              <div className="flex items-center space-x-4">
                <div className="text-2xl font-bold text-gray-800">
                  Pharma Innovation Agent
                </div>
              </div>
              <div className="border-l border-gray-300 h-12"></div>
            </div>
            <div className="text-right">
              <div className="text-gray-600 font-semibold">AI‑Powered Pharma Assistant</div>
              <div className="text-gray-500 text-sm">Advanced Pharma Research Document Intelligence</div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 w-full">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-4">
            Pharma Innovation Agent
          </h1>
          <p className="text-lg text-gray-600 max-w-3xl mx-auto">
            Upload molecule-related documents to generate professional medical summaries using advanced AI models.
          </p>
        </div>

        {!extractedText && (
          <div className="bg-white rounded-xl shadow-lg border border-gray-200 p-8">
            <FileUpload onTextExtracted={(text, quality, medical) => {
              setExtractedText(text);
              setQualityData(quality);
              setMedicalAnalysis(medical);
            }} />
          </div>
        )}

        {extractedText && (
          <div className="space-y-8">
            {/* Text Preview Toggle */}
            {extractedText && (
              <div className="bg-white rounded-lg shadow-md border border-gray-200 p-4">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-lg font-semibold text-gray-800">Extracted Text Preview</h3>
                  <button
                    onClick={() => setShowTextPreview(!showTextPreview)}
                    className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-sm font-medium"
                  >
                    {showTextPreview ? 'Hide Text' : 'Show Text'}
                  </button>
                </div>
                
                {showTextPreview && (
                  <div className="bg-gray-50 rounded-lg p-4 max-h-96 overflow-y-auto">
                    {medicalAnalysis ? (
                      <div className="space-y-4">
                        {/* Extracted Text Section - First */}
                        <div className="bg-white rounded-lg p-4 border border-gray-200">
                          <h4 className="text-sm font-semibold text-gray-800 mb-3 flex items-center">
                            <span className="w-2 h-2 bg-gray-500 rounded-full mr-2"></span>
                            Extracted Text
                          </h4>
                          <div className="text-sm text-gray-700 leading-relaxed">
                            {formatExtractedText(extractedText)}
                          </div>
                        </div>
                        
                        {/* Medical Entities Section - Second */}
                        {medicalAnalysis.medical_insights && (
                          <div className="bg-white rounded-lg p-4 border border-blue-200">
                            <h4 className="text-sm font-semibold text-blue-800 mb-3 flex items-center">
                              <span className="w-2 h-2 bg-blue-500 rounded-full mr-2"></span>
                              Medical Entities Detected
                            </h4>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
                              {medicalAnalysis.medical_insights.conditions?.length > 0 && (
                                <div>
                                  <span className="font-medium text-red-700">Conditions:</span>
                                  <div className="text-gray-600 mt-1">
                                    {medicalAnalysis.medical_insights.conditions.map((c: any, i: number) => (
                                      <span key={i} className="inline-block bg-red-50 text-red-700 px-2 py-1 rounded mr-1 mb-1">
                                        {c.Text}
                                      </span>
                                    ))}
                                  </div>
                                </div>
                              )}
                              {medicalAnalysis.medical_insights.medications?.length > 0 && (
                                <div>
                                  <span className="font-medium text-green-700">Medications:</span>
                                  <div className="text-gray-600 mt-1">
                                    {medicalAnalysis.medical_insights.medications.map((m: any, i: number) => (
                                      <span key={i} className="inline-block bg-green-50 text-green-700 px-2 py-1 rounded mr-1 mb-1">
                                        {m.Text}
                                      </span>
                                    ))}
                                  </div>
                                </div>
                              )}
                              {medicalAnalysis.medical_insights.dosages?.length > 0 && (
                                <div>
                                  <span className="font-medium text-purple-700">Dosages:</span>
                                  <div className="text-gray-600 mt-1">
                                    {medicalAnalysis.medical_insights.dosages.map((d: any, i: number) => (
                                      <span key={i} className="inline-block bg-purple-50 text-purple-700 px-2 py-1 rounded mr-1 mb-1">
                                        {d.Text}
                                      </span>
                                    ))}
                                  </div>
                                </div>
                              )}
                              {medicalAnalysis.medical_insights.procedures?.length > 0 && (
                                <div>
                                  <span className="font-medium text-blue-700">Procedures:</span>
                                  <div className="text-gray-600 mt-1">
                                    {medicalAnalysis.medical_insights.procedures.map((p: any, i: number) => (
                                      <span key={i} className="inline-block bg-blue-50 text-blue-700 px-2 py-1 rounded mr-1 mb-1">
                                        {p.Text}
                                      </span>
                                    ))}
                                  </div>
                                </div>
                              )}
                            </div>
                          </div>
                        )}
                      </div>
                    ) : (
                      <div className="text-sm text-gray-700 leading-relaxed">
                        {formatExtractedText(extractedText)}
                      </div>
                    )}
                  </div>
                )}
                
                <div className="mt-3 text-xs text-gray-500">
                  <span className="font-medium">Text Length:</span> {extractedText.length} characters | 
                  <span className="font-medium ml-2">Word Count:</span> {extractedText.split(/\s+/).length} words
                </div>
              </div>
            )}
            
            <SummaryGenerator context={extractedText} />
          </div>
        )}

        {/* Chat Section */}
        <section className="mt-10">
          <div className="bg-white rounded-xl shadow-lg border border-gray-200 p-6">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h2 className="text-2xl font-bold text-gray-800">
                  Chat with the Medical AI Assistant
                </h2>
                <p className="text-sm text-gray-600">
                  Ask questions about Pharma terminology, reports, and summaries. This tool does not replace professional medical advice.
                </p>
              </div>
            </div>

            {chatError && (
              <div className="mb-4 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-800">
                {chatError}
              </div>
            )}

            <div className="rounded-lg border border-gray-200 bg-gray-50 p-3">
              <div className="max-h-80 overflow-y-auto space-y-3 pr-1">
                {chatMessages.length === 0 ? (
                  <p className="text-sm text-gray-500">
                    Start the conversation by asking a question, for example: &quot;Help me analyze a molecule opportunity with elevated liver enzymes.&quot;
                  </p>
                ) : (
                  chatMessages.map((message, index) => (
                    <div
                      key={index}
                      className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                    >
                      <div
                        className={`max-w-[80%] rounded-lg px-3 py-2 text-sm ${
                          message.role === 'user'
                            ? 'bg-red-600 text-white'
                            : 'bg-white text-gray-800 border border-gray-200'
                        }`}
                      >
                        {message.content}
                      </div>
                    </div>
                  ))
                )}

                {isChatLoading && (
                  <div className="flex justify-start">
                    <div className="max-w-[80%] rounded-lg bg-white px-3 py-2 text-sm text-gray-600 border border-gray-200">
                      The assistant is formulating a response...
                    </div>
                  </div>
                )}
              </div>

              <form onSubmit={handleSendChat} className="mt-4 flex gap-2">
                <input
                  type="text"
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  placeholder="Type your question here..."
                  className="flex-1 rounded-lg border border-gray-300 px-3 py-2 text-sm text-gray-800 focus:border-red-500 focus:outline-none"
                  disabled={isChatLoading}
                />
                <button
                  type="submit"
                  disabled={isChatLoading || !chatInput.trim()}
                  className="rounded-lg bg-red-600 px-4 py-2 text-sm font-semibold text-white transition-colors hover:bg-red-500 disabled:cursor-not-allowed disabled:bg-red-300"
                >
                  {isChatLoading ? 'Sending…' : 'Send'}
                </button>
              </form>
            </div>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="bg-gray-800 text-white flex-shrink-0">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div>
              <h3 className="text-red-400 font-semibold mb-4">About</h3>
              <p className="text-gray-300 mb-2">AI‑Powered Pharma Assistant</p>
              <p className="text-gray-300 mb-2">Automated Molecule Opportunity Summary Generation</p>
              <p className="text-gray-300">Multi‑source Pharma Document Processing</p>
            </div>
            <div>
              <h3 className="text-red-400 font-semibold mb-4">Features</h3>
              <ul className="text-gray-300 space-y-1">
                <li>• PDF Document Processing</li>
                <li>• AI-Powered Text Extraction</li>
                <li>• Automated Summary Generation</li>
                <li>• Professional Document Formatting</li>
              </ul>
            </div>
            <div>
              <h3 className="text-red-400 font-semibold mb-4">Technology</h3>
              <ul className="text-gray-300 space-y-1">
                <li>• Advanced AI Models</li>
                <li>• Natural Language Processing</li>
                <li>• Document Intelligence</li>
                <li>• Secure Processing</li>
              </ul>
            </div>
          </div>
          <div className="border-t border-gray-700 mt-8 pt-8 text-center text-gray-400">
            <div className="flex justify-center items-center space-x-4 mb-2">
              <span>© 2025 Medical AI Assistant. All Rights Reserved.</span>
              <span>•</span>
              <span>AI-assisted clinical documentation support</span>
            </div>
            <p className="text-sm">
              This tool is for informational support only and does not provide medical advice. Always consult a licensed clinician for clinical decisions.
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;

