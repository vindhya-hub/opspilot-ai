import type { Citation } from '../types.ts'

export function CitationList({ citations }: { citations: Citation[] }) {
  if (citations.length === 0) {
    return <p className="muted">No runbook matched this ticket closely enough to cite.</p>
  }
  return (
    <div className="section">
      <h3>Sources</h3>
      <ol className="citations">
        {citations.map((citation) => (
          <li key={citation.ref}>
            <span className="ref">[{citation.ref}]</span>
            <div>
              <div className="citation-title">
                {citation.title} <span className="muted">· {citation.section}</span>
                {citation.audience === 'internal' && <span className="tag">internal</span>}
              </div>
              <p className="excerpt">{citation.excerpt}</p>
              <span className="muted small">
                kb/{citation.doc_id}.md · similarity {citation.score.toFixed(2)}
              </span>
            </div>
          </li>
        ))}
      </ol>
    </div>
  )
}
