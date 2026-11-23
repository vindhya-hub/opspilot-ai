import { useState } from 'react'

export function DraftBlock({ draft, guidance }: { draft: string; guidance: string[] }) {
  const [copied, setCopied] = useState(false)

  async function copy() {
    try {
      await navigator.clipboard.writeText(draft)
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1500)
    } catch {
      setCopied(false)
    }
  }

  return (
    <>
      <div className="section">
        <div className="section-header">
          <h3>Draft reply</h3>
          <button type="button" className="ghost small" onClick={copy}>
            {copied ? 'Copied' : 'Copy'}
          </button>
        </div>
        <pre className="draft">{draft}</pre>
      </div>
      {guidance.length > 0 && (
        <div className="section">
          <h3>Agent guidance</h3>
          <ul className="guidance">
            {guidance.map((note) => (
              <li key={note}>{note}</li>
            ))}
          </ul>
        </div>
      )}
    </>
  )
}
