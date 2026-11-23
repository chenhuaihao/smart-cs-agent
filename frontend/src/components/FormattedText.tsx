/**
 * frontend/src/components/FormattedText.tsx
 *
 * 轻量、安全（纯 React 节点，无 dangerouslySetInnerHTML）的 Markdown 渲染组件：
 *   - 供对话气泡与 Hybrid RAG 调试卡片共用；
 *   - 清洗裸露的 **粗体**、`代码`、--- 分隔线、> 引用、列表项与 Markdown 表格行。
 */

interface FormattedTextProps {
  text: string;
  compact?: boolean;
}

function renderInline(line: string) {
  // 先去掉残留的单个/未配对加粗或标题前缀，再按 **bold** 与 `code` 分段渲染
  const segments = line.split(/(\*\*[^*]+\*\*|`[^`]+`)/g);
  return segments.map((seg, idx) => {
    if (seg.startsWith('**') && seg.endsWith('**') && seg.length > 4) {
      return <strong key={idx}>{seg.slice(2, -2)}</strong>;
    }
    if (seg.startsWith('`') && seg.endsWith('`') && seg.length > 2) {
      return <code key={idx}>{seg.slice(1, -1)}</code>;
    }
    return <span key={idx}>{seg}</span>;
  });
}

function isTableSeparator(line: string): boolean {
  return /^\|?[\s:-]+\|[\s|:-]*$/.test(line);
}

export function FormattedText({ text, compact = false }: FormattedTextProps) {
  const rawLines = text.replace(/\r\n/g, '\n').split('\n');

  // 剔除首尾无意义的空行与分隔线 ---
  while (rawLines.length > 0 && (!rawLines[0].trim() || /^-{3,}$/.test(rawLines[0].trim()))) {
    rawLines.shift();
  }
  while (
    rawLines.length > 0 &&
    (!rawLines[rawLines.length - 1].trim() || /^-{3,}$/.test(rawLines[rawLines.length - 1].trim()))
  ) {
    rawLines.pop();
  }

  return (
    <div className={compact ? 'message-text message-text--compact' : 'message-text'}>
      {rawLines.map((rawLine, lineIdx) => {
        const trimmed = rawLine.trim();
        if (!trimmed) {
          return <div key={lineIdx} className="message-text__spacer" />;
        }
        if (/^-{3,}$/.test(trimmed)) {
          return <hr key={lineIdx} className="message-text__divider" />;
        }
        if (isTableSeparator(trimmed)) {
          return null;
        }
        // Markdown 表格行：| 列1 | 列2 |
        if (trimmed.startsWith('|') && trimmed.endsWith('|')) {
          const cells = trimmed
            .slice(1, -1)
            .split('|')
            .map((c) => c.trim());
          return (
            <div key={lineIdx} className="message-text__table-row">
              {cells.map((cell, cIdx) => (
                <span key={cIdx} className="message-text__table-cell">
                  {renderInline(cell)}
                </span>
              ))}
            </div>
          );
        }
        // 引用行：> 边界说明：...
        if (trimmed.startsWith('>')) {
          const content = trimmed.replace(/^>\s*/, '');
          return (
            <blockquote key={lineIdx} className="message-text__quote">
              {renderInline(content)}
            </blockquote>
          );
        }
        // 无序列表：- xxx 或 * xxx
        if (/^[-*]\s+/.test(trimmed)) {
          const content = trimmed.replace(/^[-*]\s+/, '');
          return (
            <div key={lineIdx} className="message-text__list-item">
              <span className="message-text__bullet">•</span>
              <span>{renderInline(content)}</span>
            </div>
          );
        }
        // 有序列表：1. xxx
        const orderedMatch = trimmed.match(/^(\d+\.)\s+(.+)$/);
        if (orderedMatch) {
          return (
            <div key={lineIdx} className="message-text__list-item">
              <span className="message-text__num">{orderedMatch[1]}</span>
              <span>{renderInline(orderedMatch[2])}</span>
            </div>
          );
        }
        // 标题行：### xxx
        const headingMatch = trimmed.match(/^#{1,4}\s+(.+)$/);
        if (headingMatch) {
          return (
            <p key={lineIdx} className="message-text__heading">
              {renderInline(headingMatch[1])}
            </p>
          );
        }

        return <p key={lineIdx}>{renderInline(trimmed)}</p>;
      })}
    </div>
  );
}
