/**
 * Praxis Web - Minimal Markdown Renderer
 * Renders KB content as data, not static docs. Deliberately small: headings,
 * lists, fenced/inline code, blockquotes, bold/italic, links, paragraphs.
 * Input is escaped first, so the output is safe to inject.
 */

const escapeHtml = (str) => str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');

// Inline spans: code, bold, italic, links. Applied to already-escaped text.
function renderInline(text) {
    return text
        .replace(/`([^`]+)`/g, (_, c) => `<code>${c}</code>`)
        .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
        .replace(/\*([^*]+)\*/g, '<em>$1</em>')
        .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g,
            '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
}

/**
 * Render a markdown string to an HTML string.
 * @param {string} src - Raw markdown
 * @returns {string} HTML
 */
export function renderMarkdown(src) {
    const lines = escapeHtml(src || '').split('\n');
    const out = [];
    let inCode = false;
    let listType = null; // 'ul' | 'ol' | null
    let para = [];

    const flushPara = () => {
        if (para.length) {
            out.push(`<p>${renderInline(para.join(' '))}</p>`);
            para = [];
        }
    };
    const closeList = () => {
        if (listType) { out.push(`</${listType}>`); listType = null; }
    };

    for (const line of lines) {
        const fence = line.trim().startsWith('```');
        if (fence) {
            flushPara();
            closeList();
            out.push(inCode ? '</code></pre>' : '<pre><code>');
            inCode = !inCode;
            continue;
        }
        if (inCode) { out.push(line); continue; }

        const heading = line.match(/^(#{1,6})\s+(.*)$/);
        const ul = line.match(/^\s*[-*]\s+(.*)$/);
        const ol = line.match(/^\s*\d+\.\s+(.*)$/);
        const quote = line.match(/^>\s?(.*)$/);

        if (heading) {
            flushPara(); closeList();
            const level = heading[1].length;
            out.push(`<h${level}>${renderInline(heading[2])}</h${level}>`);
        } else if (ul || ol) {
            flushPara();
            const want = ul ? 'ul' : 'ol';
            if (listType !== want) { closeList(); out.push(`<${want}>`); listType = want; }
            out.push(`<li>${renderInline((ul || ol)[1])}</li>`);
        } else if (quote) {
            flushPara(); closeList();
            out.push(`<blockquote>${renderInline(quote[1])}</blockquote>`);
        } else if (line.trim() === '') {
            flushPara(); closeList();
        } else {
            para.push(line.trim());
        }
    }
    flushPara();
    closeList();
    if (inCode) out.push('</code></pre>');
    return out.join('\n');
}

/**
 * Render a JSON blob (e.g. a run config) as a pretty-printed code block.
 * @param {string} jsonText - Raw JSON string
 * @returns {string} HTML
 */
export function renderJson(jsonText) {
    let pretty = jsonText;
    try {
        pretty = JSON.stringify(JSON.parse(jsonText), null, 2);
    } catch {
        // leave as-is if it doesn't parse
    }
    return `<pre><code>${escapeHtml(pretty)}</code></pre>`;
}
