"""Build ref/tokenizer_corpus.json: the 5 prompts plus 500 varied texts, each with
the token ids and the decoded text given by the Hugging Face tokenizer.

Usage: python ref/tokenizer_corpus.py
Needs models/hf/Qwen3-0.6B and models/wikitext/wikitext-2-raw/wiki.test.raw.
"""

import json
import random
import unicodedata
from pathlib import Path

from transformers import AutoTokenizer

REF = Path(__file__).parent
MODELS = REF.parent / "models"
N_EXTRA = 500

HAND = [
    "",
    " ",
    "  ",
    "\n",
    "\n\n",
    " \n",
    "\t",
    "a",
    " a",
    "a ",
    "a  ",
    "  a",
    "a\n",
    "a \n b",
    "a\r\nb\r\n\r\nc",
    "tabs\there\t\tand\t\t\tthere",
    "trailing spaces   ",
    "   leading spaces",
    "line one\nline two\n\nline four\n\n\nline seven",
    "mixed \t \n \r\n whitespace \n\t end",
    "Les élèves ont déjà reçu leurs cahiers à l'école, où l'on fête Noël.",
    "Ça coûte 12,50 € : c'est l'œuvre d'un naïf, sûr de lui… « Voilà ! »",
    "L'ambiguïté du mot « aiguë » gêne-t-elle le lecteur ? Peut-être.",
    "Élodie a mangé des crêpes flambées près du château de Châteauneuf-du-Pape.",
    "Straße, Übergröße, Ærøskøbing, İstanbul, Łódź, Þingvellir, señor, São Paulo",
    # Decomposed accents: the tokenizer applies NFC first.
    "café déjà vu, naïve, Ångström, ñ",
    "ḍ̇ q̣̇ ḍ̇ ą́ 각 Ω Å",
    "Привет, мир! Как дела? Всё хорошо.",
    "こんにちは世界。今日は良い天気ですね。",
    "你好，世界！今天天气怎么样？我很好，谢谢。",
    "안녕하세요 세계! 오늘 날씨가 좋네요.",
    "مرحبا بالعالم، كيف حالك اليوم؟",
    "שלום עולם, מה שלומך היום?",
    "नमस्ते दुनिया, आप कैसे हैं?",
    "สวัสดีชาวโลก วันนี้อากาศดีมาก",
    "Γειά σου Κόσμε, τι κάνεις σήμερα;",
    "Ⅷ ⅓ ② ௧௨௩ ٣٤٥ ¼ x² H₂O 10⁻³ №5",
    "I'm sure you'll say it's fine, but they've said we're late and he'd know.",
    "I'M SURE YOU'LL SAY IT'S FINE, BUT THEY'VE SAID WE'RE LATE AND HE'D KNOW.",
    "'s 't 're 've 'm 'll 'd 'S 'T 'RE 'VE 'M 'LL 'D 'x ''s rock'n'roll",
    "don't can't won't it's o'clock 'tis y'all'd've",
    "1 22 333 4444 55555 1234567890 3.14159 1,000,000 1e-9 0x1F 0b1010 1/2 50%",
    "2024-03-15T14:30:00Z +33 6 12 34 56 78 192.168.0.1 v1.2.3-rc.4 #42 $9.99",
    "a1b2c3 x86_64 utf8 sha256 mp3 4k 3D 9am 10th 21st",
    "🎉🥾⛰️🍕👨‍👩‍👧‍👦🎬🍿☔️",
    "flags 🇫🇷🇯🇵🇧🇷 skin 👍🏽👋🏿 zwj 🧑‍💻👩‍🚀 keycap 1️⃣#️⃣ misc ©️®️™️",
    "smile 😀😃😄 heart ❤️💔 text☺︎ mixed😀text 😀 spaced  😀",
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHi!<|im_end|>\n<|im_start|>assistant\n",
    "<think>\nLet me think.\n</think>\n\nThe answer is 42.<|im_end|><|endoftext|>",
    "<tool_call>\n{\"name\": \"f\", \"arguments\": {}}\n</tool_call><tool_response>ok</tool_response>",
    "<|fim_prefix|>def f():<|fim_suffix|>    return 1<|fim_middle|>",
    "almost special: <|im_start <|im_end| > <think > < think> <|unknown|> <||> <|im_start|",
    "<|im_start|><|im_start|> <|im_end|> <|endoftext|>a<|endoftext|> b",
    "[PAD151669] is not a special token for encoding",
    "for (int i = 0; i < n; ++i) { sum += a[i] * b[i]; } // dot product",
    "fn main() -> Result<(), Box<dyn Error>> {\n    println!(\"{:?}\", x?);\n    Ok(())\n}",
    "SELECT name, COUNT(*) AS n FROM users WHERE age >= 18 GROUP BY name ORDER BY n DESC;",
    "<div class=\"a b\" id='c'>\n\t<a href=\"https://example.com/?q=1&r=2#top\">link</a>\n</div>",
    "{\"key\": [1, 2.5, null, true], \"nested\": {\"a\": \"b\\n\"}}",
    "$ git commit -m \"fix: handle \\\\n\" && echo $? | grep -E '^[0-9]+$' > /dev/null 2>&1",
    "x = y**2 + 3*z - (a/b)%c ; p != q ; r <= s ; t >>= 1 ; u ||= v ; w -> x => y",
    "e = mc², ∑ᵢ xᵢ ≤ ∫₀^∞ e^{-x²} dx ≈ √π/2, ∀ε>0 ∃δ>0, α→β⇒γ",
    "!!! ??? ... --- ___ *** ### @@@ ~~~ ^^^ ``` ;;; ::: ,,, ''' \"\"\"",
    "!\n?\r\n.\n\n,\r\r-",
    "word.\nword!\n\nword?  \n  word",
    "soft­hyphen nb space zero​width thin space ideographic　space line sep",
    "MiXeD CaSe camelCase PascalCase snake_case kebab-case SCREAMING_SNAKE",
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "                                                                                ",
    "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n",
    "Supercalifragilisticexpialidocious pneumonoultramicroscopicsilicovolcanoconiosis",
    "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/tokenizer.json?download=true",
    "user.name+tag@example-domain.co.uk",
    "\x00\x01\x1f\x7f control � replacement ﻿ bom",
]

NUM_FORMATS = [
    "{a}", "{a}.{b}", "{a},{b:03d}", "-{a}", "{a}%", "${a}.{c:02d}", "{a}e{c}", "#{a}",
    "{a}/{b}", "{a}:{c:02d}", "{a} x {b} = {p}", "0x{a:X}", "{a}km", "v{c}.{d}.{a}",
    "+{c} ({b}) {a}", "{a}th", "{a}°C", "{a}–{b}", "({a})", "[{a}, {b}, {p}]",
]
EMOJIS = "😀🎉🍕🚀❤️👍🏽🤖🌍🔥✨🥾⛰️👨‍👩‍👧‍👦🇫🇷☔️🧑‍💻"
SPACES = ["  ", "   ", "\t", "\n", " \n", "\n ", " \n ", "\r\n", "\n\n", " \t "]


def main():
    rng = random.Random(0)
    wiki = (MODELS / "wikitext/wikitext-2-raw/wiki.test.raw").read_text().splitlines()
    wiki = [line[:300] for line in wiki if len(line) > 60]
    rng.shuffle(wiki)
    code = (REF / "dump.py").read_text().splitlines(keepends=True)

    texts = list(HAND)
    for i in range(0, 120, 4):  # 30 code chunks, indentation and newlines included
        texts.append("".join(code[i : i + 4]))
    for _ in range(50):  # numbers
        parts = []
        for _ in range(rng.randint(2, 6)):
            a, b = rng.randint(0, 10**rng.randint(1, 9)), rng.randint(0, 999)
            c, d = rng.randint(0, 59), rng.randint(0, 9)
            parts.append(rng.choice(NUM_FORMATS).format(a=a, b=b, c=c, d=d, p=a * b))
        texts.append(rng.choice([" ", ", ", "; ", "\n"]).join(parts))
    for _ in range(100):  # multiple spaces and mixed whitespace
        words = wiki.pop().split(" ")
        texts.append(
            "".join(w + (rng.choice(SPACES) if rng.random() < 0.3 else " ") for w in words)
        )
    emojis = list(EMOJIS)
    for _ in range(50):  # emojis inside text, with and without spaces
        words = wiki.pop().split(" ")[:20]
        for _ in range(rng.randint(1, 5)):
            words.insert(rng.randrange(len(words) + 1), "".join(rng.choices(emojis, k=rng.randint(1, 3))))
        texts.append(rng.choice([" ", ""]).join(words))
    marks = [chr(c) for c in range(0x300, 0x370)] + list("\u0591\u05b0\u064b\u0651\u093c\u094d\u0e38\u1dc0\u20d0\u3099")
    bases = list("aeinouAEOUycsz \u00e9\u00fc\u1e0b\u0391\u03b9\u0438\u05d0\u0627\u0915\u304b\u30cf\u1100\u1161\u11a8\uac00\ud55c\u2126\u212b\u0344\u0f73")
    for _ in range(40):  # random combining sequences: exercises NFC reordering and composition
        texts.append("".join(
            rng.choice(bases) + "".join(rng.choices(marks, k=rng.choice([0, 0, 1, 1, 2, 3])))
            for _ in range(rng.randint(5, 30))
        ))
    while len(texts) < N_EXTRA:  # plain prose
        texts.append(wiki.pop())
    assert len(texts) == N_EXTRA and len(set(texts)) == N_EXTRA

    prompts = [p["text"] for p in json.loads((REF / "prompts.json").read_text())]
    tok = AutoTokenizer.from_pretrained(MODELS / "hf/Qwen3-0.6B")
    corpus = []
    for text in prompts + texts:
        ids = tok(text).input_ids
        decoded = tok.decode(ids)
        assert decoded == unicodedata.normalize("NFC", text), repr(text)
        corpus.append({"text": text, "ids": ids, "decoded": decoded})
    (REF / "tokenizer_corpus.json").write_text(
        json.dumps(corpus, ensure_ascii=True, indent=0) + "\n"
    )
    print(f"{len(corpus)} texts, {sum(len(c['ids']) for c in corpus)} tokens")


if __name__ == "__main__":
    main()
