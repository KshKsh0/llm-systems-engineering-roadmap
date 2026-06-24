from pathlib import Path
import re

RAW_PATH = Path('../data/raw/sample.txt')
CLEAN_PATH = Path('../data/clean/clean.txt')


def clean_text(text ) ->str:
    
    text  = text.replace('\r\n' , '\n')
    text = re.sub(r'\n{3,}' , '\n\n' , text)
    text = re.sub(r'[ \t]+' , ' ' , text)
    return text.strip()


def deduplicate(text:str )->str:
    
    seen = set()
    lines= []
    for line in text.splitlines():
        normalized= line.strip().lower()
        
        if not normalized:
            lines.append('')
            continue
        if normalized not in seen:
            seen.add(normalized)
            lines.append(line.strip())
        
    return '\n'.join(lines)
            
            
def main():
    
    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw_text = RAW_PATH.read_text(encoding='utf-8')
    cleaned = clean_text(text=raw_text)
    dedup = deduplicate(cleaned)
    CLEAN_PATH.write_text(dedup)
    print(f'saved the cleaned version {CLEAN_PATH} !')
    
    
if __name__ == '__main__':
    main()