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
            '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>')
        // Internal wiki edges: a relative .md target is another KB node; the
        // click handler resolves the stem to its note/doc id and opens it.
        .replace(/\[([^\]]+)\]\(([\w./-]+\.md)\)/g, (_, label, path) => {
            const stem = path.split('/').pop().replace(/\.md$/, '');
            return `<a href="#" class="kb-wiki-link" data-kb-stem="${stem}">${label}</a>`;
        })
        // Source references are wiki edges too: a praxis/ path (bare or in a
        // code span) opens that file's KB entry.
        .replace(/(<code>)?(praxis\/[\w/]+\.py)(<\/code>)?/g,
            (_, o, path, c) => `${o || ''}<a href="#" class="kb-wiki-link" data-kb-stem="${path}">${path}</a>${c || ''}`)
        // Sibling-repo references (../platformer) have no node of their own;
        // the wiki resolver falls through to a search on the name, which
        // gathers everything that cites it.
        .replace(/(<code>)?\.\.\/([\w-]+)(<\/code>)?/g,
            (_, o, name, c) => `${o || ''}<a href="#" class="kb-wiki-link" data-kb-stem="${name}">../${name}</a>${c || ''}`);
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

// Python keywords for the lightweight highlighter.
const PY_KEYWORDS = new Set(('and as assert async await break class continue def del elif else except ' +
    'finally for from global if import in is lambda nonlocal not or pass raise ' +
    'return try while with yield None True False self').split(' '));

/**
 * Render a source file as a single highlighted code block. No grammar - just
 * token-level regexes over escaped text, enough to give the page shape.
 * @param {string} src - Raw file text
 * @returns {string} HTML
 */
export function renderCode(src) {
    const html = escapeHtml(src || '').replace(
        // one pass: strings | comments | numbers | words
        /(&quot;{3}[\s\S]*?&quot;{3}|'{3}[\s\S]*?'{3}|&quot;[^&\n]*?&quot;|'[^'\n]*')|(#[^\n]*)|(\b\d[\d_.]*\b)|([A-Za-z_]\w*)/g,
        (m, str, comment, num, word) => {
            if (str) return `<span class="tok-str">${str}</span>`;
            if (comment) return `<span class="tok-com">${comment}</span>`;
            if (num) return `<span class="tok-num">${num}</span>`;
            if (word && PY_KEYWORDS.has(word)) return `<span class="tok-kw">${word}</span>`;
            return m;
        }
    );
    return `<pre class="code-file"><code>${html}</code></pre>`;
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
