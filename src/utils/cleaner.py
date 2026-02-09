import re

class MarkdownCleaner:
    def __init__(self):
        # 匹配Abstract
        self.head_pattern = re.compile(
            r'(^|\n)#+\s*Abstract', 
            re.IGNORECASE
        )

        # 1. 匹配尾部
        self.ref_pattern = re.compile(
            r'(^|\n)#+\s*(References?|Bibliography|Cited Works|Acknowledg[e]?ments?|Funding|Appendix)(.*)', 
            re.IGNORECASE | re.DOTALL
        )
        # 2. HTML 表格
        self.table_pattern = re.compile(r'<table[^>]*>.*?</table>', re.IGNORECASE | re.DOTALL)
        # 3. 图片标签
        self.img_tag_pattern = re.compile(r'^\s*!\[.*?\]\(.*?\)\s*$')
        # 4. Caption
        self.caption_pattern = re.compile(
            r'^\s*(Figure|Fig\.|Table)\s*\d+(?:\s*[:.]|\s*$)', 
            re.IGNORECASE
        )
        # 5. 纯页码
        self.page_number_pattern = re.compile(r'^\s*\d+\s*$')

    def remove_head_sections(self, text):
        """
        删除 Abstract 之前的所有内容
        """
        match = self.head_pattern.search(text)
        if match:
            return text[match.start():].lstrip()
        return text

    def remove_tail_sections(self, text):
        match = self.ref_pattern.search(text)
        if match: return text[:match.start()].strip()
        return text

    def remove_html_tables(self, text):
        return self.table_pattern.sub('', text)

    def clean_inline_images(self, line):
        return re.sub(r'!\[.*?\]\(.*?\)', '', line)

    def should_merge(self, prev_text, current_line):
        if not prev_text: return False
        prev = prev_text.strip()
        curr = current_line.strip()
        
        # 1. Markdown 标题 (#) 永远不合并
        if curr.startswith('#') or prev.startswith('#'): return False

        is_curr_bullet = re.match(r'^[-*]\s', curr)
        
        if is_curr_bullet:
            # 规则 A: 上一行以冒号结尾 -> 合并 (e.g., "Methods:")
            if prev.endswith(':'):
                return True
            
            # 规则 B: 连续列表合并
            # 如果上一行已经是 "Text: - Item 1" 这种形式，
            # 或者上一行本身就是 "- Item 1" ，需要检测上一行是否处于“列表模式”。
            if re.search(r':\s*[-*]\s', prev):
                return True

            prev_is_bullet = re.match(r'^[-*]\s', prev)
            prev_is_number = re.match(r'^\d+\.\s', prev)
            if prev_is_bullet or prev_is_number:
                return True

        # 2. 列表合并逻辑 (处理数字列表的情况)
        is_number = re.match(r'^\d+\.\s', curr)
        if is_number:
            prev_is_bullet = re.match(r'^[-*]\s', prev)
            prev_is_number = re.match(r'^\d+\.\s', prev)
            if prev_is_bullet or prev_is_number:
                return True

        # 3. 连字符
        if prev.endswith('-'): return True
        # 4. 强标点
        if prev.endswith((',', ':', ';')): return True
        # 5. 公式块
        if prev.endswith('$$') or curr.startswith('$$'): return True
        # 6. 行内公式
        if prev.endswith('$') and not prev.endswith('$$'):
             if curr and (curr[0].islower() or curr[0] in [',', '.', ';']): return True
        # 7. 小写开头
        if curr and curr[0].islower(): return True
        # 8. 兜底
        if not prev.endswith(('.', '!', '?', '”', '"')): return True

        return False

    def pre_mark_garbage(self, lines):
        is_garbage = [False] * len(lines)
        img_indices = []
        cap_indices = []
        
        # 阈值，超过这个长度的行，哪怕以 Figure: 开头，也认为是正文
        MAX_CAPTION_LENGTH = 1000

        for i, line in enumerate(lines):
            line_strip = line.strip()
            if not line_strip: continue 

            # 1. 图片标签 
            if self.img_tag_pattern.match(line_strip):
                is_garbage[i] = True
                img_indices.append(i)
            
            # 2. Caption
            elif self.caption_pattern.match(line_strip):
                # 如果这一行太长, 即使它符合正则，默认它是正文，不删。
                if len(line_strip) < MAX_CAPTION_LENGTH:
                    is_garbage[i] = True
                    cap_indices.append(i)
                else:
                    pass

            elif self.page_number_pattern.match(line_strip):
                is_garbage[i] = True

        for cap_idx in cap_indices:
            # 向前找图片
            for look_back in range(1, 5): 
                img_idx = cap_idx - look_back
                if img_idx < 0: break
                if is_garbage[img_idx] and self.img_tag_pattern.match(lines[img_idx].strip()):
                    for k in range(img_idx + 1, cap_idx): is_garbage[k] = True
                    break
            # 向后找图片
            for look_forward in range(1, 5):
                img_idx = cap_idx + look_forward
                if img_idx >= len(lines): break
                if is_garbage[img_idx] and self.img_tag_pattern.match(lines[img_idx].strip()):
                    for k in range(cap_idx + 1, img_idx): is_garbage[k] = True
                    break
        return is_garbage

