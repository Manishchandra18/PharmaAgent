// frontend/src/components/SummaryGenerator.tsx

import { useState } from "react";
import { Sparkles, LoaderCircle, Edit3, Eye, FileText, FileDown } from "lucide-react";

interface SummaryGeneratorProps {
  context: string; // Extracted text from regulatory/clinical/pharma documents
}

// Logo component for Pharma Innovation Assistant
function PharmaLogo() {
  return (
    <div className="absolute top-4 right-4 text-right">
      <div className="text-gray-600 font-semibold text-sm">
        Pharma Innovation Assistant
      </div>
    </div>
  );
}

// Component to render the molecule innovation summary
function DocumentRenderer({ content }: { content: string }) {
  const renderFormattedContent = (text: string) => {
    let cleanedText = text
      .replace(/={3,}/g, '')
      .replace(/\*\*/g, '')
      .replace(/\*([^*]+)\*/g, '$1')
      .replace(/^MOLECULE INNOVATION SUMMARY\s*$/gm, '')
      .replace(/^.*MOLECULE INNOVATION SUMMARY.*$/gm, '')
      .replace(/MOLECULE INNOVATION SUMMARY/g, '');

    const sections = cleanedText.split(/(?=### )/);

    return sections.map((section, index) => {
      if (!section.trim()) return null;
      const lines = section.split('\n');
      const header = lines[0];
      const content = lines.slice(1).join('\n');

      if (content.includes('|') && content.includes('---')) {
        return renderTableSection(header, content, index);
      }

      if (header.startsWith('### HEADER SECTION')) {
        return renderHeaderSection(header, content, index);
      }

      return renderRegularSection(header, content, index);
    });
  };

  const renderHeaderSection = (_header: string, content: string, index: number) => {
    const lines = content.split('\n').filter(line => line.trim());
    const portfolioDetails: string[] = [];
    const moleculeDetails: string[] = [];
    let currentSection = '';

    lines.forEach(line => {
      if (line.includes('Portfolio / Organization Details')) {
        currentSection = 'portfolio';
      } else if (line.includes('Molecule Details')) {
        currentSection = 'molecule';
      } else if (currentSection) {
        if (currentSection === 'portfolio') portfolioDetails.push(line.trim());
        if (currentSection === 'molecule') moleculeDetails.push(line.trim());
      }
    });

    return (
      <div key={index} className="mb-8 p-6 bg-white rounded-lg shadow-md relative">
        <PharmaLogo />
        <h1 className="text-3xl font-bold text-gray-800 mb-6 text-center pb-3">
          MOLECULE INNOVATION SUMMARY
        </h1>

        <div className="space-y-8">
          {/* Portfolio Details */}
          <div className="space-y-3">
            <h3 className="text-lg font-semibold text-gray-800 border-b border-gray-300 pb-1">
              Portfolio / Organization Details
            </h3>
            {portfolioDetails.length > 0 ? (
              portfolioDetails.map((detail, i) => (
                <div key={i} className="text-sm text-gray-700">
                  {renderTextWithPlaceholders(detail)}
                </div>
              ))
            ) : (
              <div className="text-sm text-gray-500 italic">Not Documented</div>
            )}
          </div>

          {/* Molecule Details */}
          <div className="space-y-3">
            <h3 className="text-lg font-semibold text-gray-800 border-b border-gray-300 pb-1">
              Molecule Details
            </h3>
            {moleculeDetails.length > 0 ? (
              moleculeDetails.map((detail, i) => (
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
                  <span className="text-red-500 mr-2 mt-1 font-semibold">
                    {line.match(/^\d+\./)?.[0]}
                  </span>
                  <span>{renderTextWithPlaceholders(line.replace(/^\d+\.\s*/, ''))}</span>
                </div>
              );
            } else {
              return (
                <div key={lineIndex}>{renderTextWithPlaceholders(line)}</div>
              );
            }
          })}
        </div>
      </div>
    );
  };

  const renderTextWithPlaceholders = (text: string) => {
    const processedText = text.replace(/\[([^\]]*)\]/g, (_, content) => content);
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
  const [summaryHeight] = useState<number>(40);

  const handleGenerateSummary = async () => {
    setIsLoading(true);
    setError(null);
    setSummary(null);
    setEditedSummary(null);
    setIsEditing(false);

    try {
      const response = await fetch('http://localhost:8000/api/v1/generate-summary', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ context })
      });

      if (!response.ok) throw new Error("Failed to generate summary");

      const data = await response.json();
      setSummary(data.summary);
      setEditedSummary(data.summary);

    } catch (err) {
      setError(err instanceof Error ? err.message : "An unknown error occurred.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleDownloadDOCX = async () => {
    const contentToDownload = editedSummary || summary;
    if (!contentToDownload) return;

    try {
      const response = await fetch('http://localhost:8000/api/v1/download-docx', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: contentToDownload }),
      });

      if (!response.ok) throw new Error('Failed to generate DOCX');

      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = 'molecule_summary.docx';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);

    } catch (error) {
      alert('DOCX download failed.');
    }
  };

  const currentContent = editedSummary || summary;

  return (
    <div className="bg-white p-6 rounded-xl shadow-lg border border-gray-200">
      <div className="flex flex-col sm:flex-row justify-between items-center mb-6">
        <h2 className="text-2xl font-bold text-gray-800 mb-4 sm:mb-0">
          Molecule Innovation Summary
        </h2>

        <div className="flex gap-2">
          <button
            onClick={handleGenerateSummary}
            disabled={isLoading}
            className="bg-red-600 hover:bg-red-500 text-white font-bold py-2 px-4 rounded-lg"
          >
            {isLoading ? <LoaderCircle className="animate-spin mr-2" /> : <Sparkles className="mr-2" />}
            {summary ? "Regenerate Summary" : "Generate Summary"}
          </button>

          <button
            onClick={() => window.location.reload()}
            className="bg-orange-600 hover:bg-orange-500 text-white font-bold py-2 px-4 rounded-lg"
          >
            New File
          </button>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="my-4 text-center text-red-700 bg-red-50 border border-red-200 p-3 rounded-lg">
          {error}
        </div>
      )}

      {/* Summary */}
      {currentContent && (
        <>
          <div className="mt-4 p-4 bg-gray-50 rounded-lg border border-gray-200">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold text-gray-800">
                {isEditing ? "Edit Summary" : "Generated Summary"}
              </h3>

              {!isEditing && (
                <div className="flex gap-2 items-center">
                  <button
                    onClick={() => setViewMode('formatted')}
                    className={`px-3 py-1 rounded ${viewMode === 'formatted' ? 'bg-red-600 text-white' : 'bg-gray-200'}`}
                  >
                    <Eye className="mr-1 h-4 w-4" /> Formatted
                  </button>

                  <button
                    onClick={() => setViewMode('raw')}
                    className={`px-3 py-1 rounded ${viewMode === 'raw' ? 'bg-red-600 text-white' : 'bg-gray-200'}`}
                  >
                    <FileText className="mr-1 h-4 w-4" /> Raw
                  </button>

                  <button
                    onClick={() => setIsEditing(true)}
                    className="bg-orange-600 hover:bg-orange-500 text-white font-bold py-2 px-4 rounded-lg"
                  >
                    <Edit3 className="mr-2 h-4 w-4" /> Edit
                  </button>
                </div>
              )}
            </div>
          </div>

          {isEditing ? (
            <textarea
              value={editedSummary || ""}
              onChange={(e) => setEditedSummary(e.target.value)}
              className="w-full h-96 p-4 bg-white text-gray-800 border border-gray-300 rounded-lg font-mono text-sm"
              placeholder="Edit the molecule summary here..."
            />
          ) : viewMode === 'formatted' ? (
            <div className="overflow-y-auto border border-slate-600 rounded-lg bg-white" style={{ height: `${summaryHeight}vh` }}>
              <DocumentRenderer content={currentContent} />
            </div>
          ) : (
            <pre className="p-4 bg-white border border-gray-300 rounded-lg text-gray-800 whitespace-pre-wrap">
              {currentContent}
            </pre>
          )}

          <div className="flex justify-end gap-2 mt-4">
            <button
              onClick={handleDownloadDOCX}
              className="bg-red-600 hover:bg-red-500 text-white font-bold py-2 px-4 rounded-lg"
            >
              <FileDown className="mr-2 h-5 w-5" /> Download Summary
            </button>
          </div>
        </>
      )}
    </div>
  );
}
