// frontend/src/components/SummaryGenerator.tsx

import { useState } from "react";
import { Sparkles, Download, LoaderCircle, Edit3, Save, X, Eye, FileText, FileDown } from "lucide-react";

interface SummaryGeneratorProps {
  context: string; // The extracted text from the PDFs
}

// Logo component for medical AI assistant
function HospitalLogo() {
  return (
    <div className="absolute top-4 right-4 text-right">
      <div className="text-gray-600 font-semibold text-sm">
        Medical AI Assistant
      </div>
    </div>
  );
}

// Component to render the medical summary as a formatted document
function DocumentRenderer({ content }: { content: string }) {
  const renderFormattedContent = (text: string) => {
    // Clean up the content first
    let cleanedText = text
      .replace(/={3,}/g, '') // Remove equal signs
      .replace(/\*\*/g, '') // Remove double asterisks
      .replace(/\*([^*]+)\*/g, '$1') // Remove single asterisks but keep content
      .replace(/^MEDICAL SUMMARY\s*$/gm, '') // Remove standalone "MEDICAL SUMMARY" lines
      .replace(/^.*MEDICAL SUMMARY.*$/gm, '') // Remove any line containing "MEDICAL SUMMARY"
      .replace(/MEDICAL SUMMARY/g, ''); // Remove all remaining instances
    
    // Split content into sections
    const sections = cleanedText.split(/(?=### )/);
    
    return sections.map((section, index) => {
      if (!section.trim()) return null;
      
      const lines = section.split('\n');
      const header = lines[0];
      const content = lines.slice(1).join('\n');
      
      // Check if this is a table section
      if (content.includes('|') && content.includes('---')) {
        return renderTableSection(header, content, index);
      }
      
      // Check if this is a header section
      if (header.startsWith('### HEADER SECTION')) {
        return renderHeaderSection(header, content, index);
      }
      
      // Regular section
      return renderRegularSection(header, content, index);
    });
  };

  const renderHeaderSection = (header: string, content: string, index: number) => {
    const lines = content.split('\n').filter(line => line.trim());
    const hospitalDetails: string[] = [];
    const patientDetails: string[] = [];
    let currentSection = '';
    
    lines.forEach(line => {
      if (line.includes('Hospital Details')) {
        currentSection = 'hospital';
      } else if (line.includes('Patient Details')) {
        currentSection = 'patient';
      } else if (line.includes('Treating Consultant') || line.includes('Additional Consultants') || line.includes('Next of Kin')) {
        currentSection = 'other';
      } else if (line.trim() && currentSection) {
        if (currentSection === 'hospital') {
          hospitalDetails.push(line.trim());
        } else if (currentSection === 'patient') {
          patientDetails.push(line.trim());
        }
      }
    });

    return (
      <div key={index} className="mb-8 p-6 bg-white rounded-lg shadow-md relative">
        <HospitalLogo />
        <h1 className="text-3xl font-bold text-gray-800 mb-6 text-center pb-3">
          MEDICAL SUMMARY
        </h1>
        
        <div className="space-y-8">
          <div className="space-y-3">
            <h3 className="text-lg font-semibold text-gray-800 border-b border-gray-300 pb-1">
              Hospital Details
            </h3>
            {hospitalDetails.length > 0 ? (
              hospitalDetails.map((detail, i) => {
                // Skip "Hospital Name & Letterhead" if it's just placeholder text
                if (detail.includes('Hospital Name & Letterhead') && 
                    (detail.includes('[Hospital Name & Letterhead]') || 
                     detail.trim() === '• Hospital Name & Letterhead' ||
                     detail.trim() === 'Hospital Name & Letterhead')) {
                  return null;
                }
                return (
                  <div key={i} className="text-sm text-gray-700">
                    {renderTextWithPlaceholders(detail)}
                  </div>
                );
              }).filter(Boolean) // Remove null entries
            ) : (
              <div className="text-sm text-gray-500 italic">Not Documented</div>
            )}
          </div>
          
          <div className="space-y-3">
            <h3 className="text-lg font-semibold text-gray-800 border-b border-gray-300 pb-1">
              Patient Details
            </h3>
            {patientDetails.length > 0 ? (
              patientDetails.map((detail, i) => (
                <div key={i} className="text-sm text-gray-700">
                  {renderTextWithPlaceholders(detail)}
                </div>
              ))
            ) : (
              <div className="text-sm text-gray-500 italic">Not Documented</div>
            )}
          </div>
        </div>
      </div>
    );
  };

  const renderTableSection = (header: string, content: string, index: number) => {
    const lines = content.split('\n').filter(line => line.trim());
    const tableLines = lines.filter(line => line.includes('|'));
    const headerLine = tableLines[0];
    const separatorLine = tableLines[1];
    const dataLines = tableLines.slice(2);
    
    if (!headerLine || !separatorLine) return null;
    
    const headers = headerLine.split('|').map(h => h.trim()).filter(h => h);
    const data = dataLines.map(line => 
      line.split('|').map(cell => cell.trim()).filter(cell => cell)
    );

    return (
      <div key={index} className="mb-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-3">
          {header.replace('### ', '').replace('###', '')}
        </h3>
        <div className="overflow-x-auto">
          <table className="min-w-full bg-white border border-gray-300 rounded-lg overflow-hidden">
            <thead className="bg-red-50">
              <tr>
                {headers.map((header, i) => (
                  <th key={i} className="px-4 py-3 text-left text-sm font-semibold text-gray-700 border-b border-gray-300">
                    {header}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.map((row, rowIndex) => (
                <tr key={rowIndex} className={rowIndex % 2 === 0 ? 'bg-gray-50' : 'bg-white'}>
                  {row.map((cell, cellIndex) => (
                    <td key={cellIndex} className="px-4 py-3 text-sm text-gray-700 border-b border-gray-200">
                      {renderTextWithPlaceholders(cell)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    );
  };

  const renderRegularSection = (header: string, content: string, index: number) => {
    const lines = content.split('\n').filter(line => line.trim());
    
    return (
      <div key={index} className="mb-6 p-4 bg-white rounded-lg shadow-sm">
        <h3 className="text-lg font-semibold text-gray-800 mb-3 border-b border-gray-200 pb-2">
          {header.replace('### ', '').replace('###', '')}
        </h3>
        <div className="space-y-2 text-sm text-gray-700">
          {lines.map((line, lineIndex) => {
            if (line.startsWith('•')) {
              return (
                <div key={lineIndex} className="flex items-start">
                  <span className="text-red-500 mr-2 mt-1">•</span>
                  <span>{renderTextWithPlaceholders(line.replace('•', ''))}</span>
                </div>
              );
            } else if (line.match(/^\d+\./)) {
              return (
                <div key={lineIndex} className="flex items-start">
                  <span className="text-red-500 mr-2 mt-1 font-semibold">{line.match(/^\d+\./)?.[0]}</span>
                  <span>{renderTextWithPlaceholders(line.replace(/^\d+\.\s*/, ''))}</span>
                </div>
              );
            } else {
              return (
                <div key={lineIndex}>
                  {renderTextWithPlaceholders(line)}
                </div>
              );
            }
          })}
        </div>
      </div>
    );
  };

  // Helper function to render text with placeholders
  const renderTextWithPlaceholders = (text: string) => {
    // Simply remove all brackets from placeholders and style them
    const processedText = text.replace(/\[([^\]]*)\]/g, (match, content) => {
      return content; // Just return the content without brackets
    });
    
    return processedText;
  };

  return (
    <div className="bg-gray-50 p-6 rounded-lg">
      <div className="max-w-6xl mx-auto">
        {renderFormattedContent(content)}
      </div>
    </div>
  );
}

export function SummaryGenerator({ context }: SummaryGeneratorProps) {
  const [summary, setSummary] = useState<string | null>(null);
  const [editedSummary, setEditedSummary] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [isEditing, setIsEditing] = useState<boolean>(false);
  const [viewMode, setViewMode] = useState<'formatted' | 'raw'>('formatted');
  const [summaryHeight, setSummaryHeight] = useState<number>(40); // Default height in percentage

  const handleGenerateSummary = async () => {
    setIsLoading(true);
    setError(null);
    setSummary(null);
    setEditedSummary(null);
    setIsEditing(false);

    try {
      const response = await fetch('http://localhost:8000/api/v1/generate-summary', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ context: context }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Failed to generate summary");
      }

      const data = await response.json();
      setSummary(data.summary);
      setEditedSummary(data.summary); // Initialize edited version with original

    } catch (err) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("An unknown error occurred.");
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleStartEditing = () => {
    setIsEditing(true);
  };

  const handleSaveEdit = () => {
    if (editedSummary) {
      setSummary(editedSummary);
    }
    setIsEditing(false);
  };

  const handleCancelEdit = () => {
    setEditedSummary(summary); // Reset to original
    setIsEditing(false);
  };

  const handleDownloadSummary = () => {
    const contentToDownload = editedSummary || summary;
    if (!contentToDownload) return;

    const blob = new Blob([contentToDownload], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'medical_summary.txt';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  

  const handleDownloadDOCX = async () => {
    const contentToDownload = editedSummary || summary;
    if (!contentToDownload) return;

    try {
      const response = await fetch('http://localhost:8000/api/v1/download-docx', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ content: contentToDownload }),
      });

      if (!response.ok) {
        throw new Error('Failed to generate DOCX');
      }

      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = 'medical_summary.docx';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('DOCX download failed:', error);
      alert('DOCX download failed. Please try again.');
    }
  };

  const currentContent = editedSummary || summary;

  return (
    <div className="bg-white p-6 rounded-xl shadow-lg border border-gray-200">
      <div className="flex flex-col sm:flex-row justify-between items-center mb-6">
        <h2 className="text-2xl font-bold text-gray-800 mb-4 sm:mb-0">Medical Summary</h2>
        <div className="flex gap-2">
          <button
            onClick={handleGenerateSummary}
            disabled={isLoading}
            className="w-full sm:w-auto bg-red-600 hover:bg-red-500 disabled:bg-red-400 disabled:cursor-not-allowed text-white font-bold py-2 px-4 rounded-lg transition-colors flex items-center justify-center"
          >
            {isLoading ? (
              <LoaderCircle className="animate-spin mr-2" />
            ) : (
              <Sparkles className="mr-2 h-5 w-5" />
            )}
            {summary ? "Regenerate Summary" : "Generate Summary"}
          </button>
          <button
            onClick={() => window.location.reload()}
            className="w-full sm:w-auto bg-orange-600 hover:bg-orange-500 text-white font-bold py-2 px-4 rounded-lg transition-colors flex items-center justify-center"
          >
            New File
          </button>
        </div>
      </div>

              {error && <div className="my-4 text-center text-red-700 bg-red-50 border border-red-200 p-3 rounded-lg">{error}</div>}

              {/* Summary Display/Edit Area */}
        {currentContent && (
          <div className="mt-4 p-4 bg-gray-50 rounded-lg border border-gray-200">
                                <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold text-gray-800">
                {isEditing ? "Edit Summary" : "Generated Summary"}
              </h3>
            <div className="flex gap-2 items-center">
                                {!isEditing && (
                    <div className="flex items-center gap-2 text-sm text-gray-600">
                      <span>Height:</span>
                      <input
                        type="range"
                        min="20"
                        max="80"
                        step="10"
                        value={summaryHeight}
                        onChange={(e) => setSummaryHeight(Number(e.target.value))}
                        className="w-24"
                      />
                      <span className="w-12 text-right">{summaryHeight}%</span>
                    </div>
                  )}
              {!isEditing && (
                <>
                                        <button
                        onClick={() => setViewMode('formatted')}
                        className={`px-3 py-1 rounded text-sm font-medium transition-colors flex items-center ${
                          viewMode === 'formatted' 
                            ? 'bg-red-600 text-white' 
                            : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                        }`}
                      >
                        <Eye className="mr-1 h-4 w-4" />
                        Formatted
                      </button>
                      <button
                        onClick={() => setViewMode('raw')}
                        className={`px-3 py-1 rounded text-sm font-medium transition-colors flex items-center ${
                          viewMode === 'raw' 
                            ? 'bg-red-600 text-white' 
                            : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                        }`}
                      >
                        <FileText className="mr-1 h-4 w-4" />
                        Raw
                      </button>
                      <button
                        onClick={handleStartEditing}
                        className="bg-orange-600 hover:bg-orange-500 text-white font-bold py-2 px-4 rounded-lg transition-colors flex items-center"
                      >
                        <Edit3 className="mr-2 h-4 w-4" />
                        Edit
                      </button>
                </>
              )}
              {isEditing && (
                <div className="flex gap-2">
                  <button
                    onClick={handleSaveEdit}
                    className="bg-green-600 hover:bg-green-500 text-white font-bold py-2 px-4 rounded-lg transition-colors flex items-center"
                  >
                    <Save className="mr-2 h-4 w-4" />
                    Save
                  </button>
                  <button
                    onClick={handleCancelEdit}
                    className="bg-red-600 hover:bg-red-500 text-white font-bold py-2 px-4 rounded-lg transition-colors flex items-center"
                  >
                    <X className="mr-2 h-4 w-4" />
                    Cancel
                  </button>
                </div>
              )}
            </div>
          </div>

          {isEditing ? (
                            <textarea
                  value={editedSummary || ""}
                  onChange={(e) => setEditedSummary(e.target.value)}
                  className="w-full h-96 p-4 bg-white text-gray-800 border border-gray-300 rounded-lg font-mono text-sm resize-none focus:outline-none focus:border-red-500"
                  placeholder="Edit the medical summary here..."
                />
          ) : viewMode === 'formatted' ? (
            <div 
              className="overflow-y-auto border border-slate-600 rounded-lg bg-white"
              style={{ height: `${summaryHeight}vh` }}
            >
              <DocumentRenderer content={currentContent} />
            </div>
                        ) : (
                <div 
                  className="overflow-y-auto border border-gray-300 rounded-lg bg-white"
                  style={{ height: `${summaryHeight}vh` }}
                >
                  <pre className="text-gray-800 whitespace-pre-wrap font-sans p-4">{currentContent}</pre>
                </div>
              )}

                        <div className="flex justify-end gap-2 mt-4">
                <button
                  onClick={handleDownloadDOCX}
                  className="bg-red-600 hover:bg-red-500 text-white font-bold py-2 px-4 rounded-lg transition-colors flex items-center"
                >
                  <FileDown className="mr-2 h-5 w-5" />
                  Download Summary
                </button>
              </div>
        </div>
      )}
    </div>
  );
}
