You read one turn of a phone call to a food delivery support line in India, as speech-to-text wrote it. The caller speaks Hindi, English or both, written in English letters.

Find every number the caller says. A number can be digits (4521) or words, in Hindi or English, said one digit at a time or in pairs: "chaar paanch do ek", "four five two one", "paintaalis ikkees", "chaar hazaar paanch sau ikkees".

List the numbers in the order they are said. For each, copy only its number words exactly as written, leaving out any other words said in between, and give it one tag:
- order: a number that could be the caller's order number, usually 4 digits, with or without the word "order"
- other: an amount, a price, a time, an address, a count, or any other number
- unsure: an order number the caller isn't sure of, or takes back
Give an empty list only when the caller says no number at all.

Examples:
"teen aath do chhah" -> [{"words": "teen aath do chhah", "tag": "order"}]
"Haan ji, chhattees baarah, Priya Sharma" -> [{"words": "chhattees baarah", "tag": "order"}]
"6181 nahi, six one eight two" -> [{"words": "6181", "tag": "order"}, {"words": "six one eight two", "tag": "order"}]
"Mere do order hain, 3355 aur teen teen paanch chhah" -> [{"words": "3355", "tag": "order"}, {"words": "teen teen paanch chhah", "tag": "order"}]
"Order 2718 par chaar sau pachaas rupaye wapas chahiye" -> [{"words": "2718", "tag": "order"}, {"words": "chaar sau pachaas", "tag": "other"}]
"Do aath saat ek, galat hai, ruko dekhta hoon" -> [{"words": "Do aath saat ek", "tag": "unsure"}]
"Order saat hmm do, ek sec, nau chaar hai" -> [{"words": "saat do nau chaar", "tag": "order"}]
"Order 38 aa 67 kab aayega" -> [{"words": "38 67", "tag": "order"}]
"3167 oh no, sorry three thousand 16 aath" -> [{"words": "3167", "tag": "unsure"}, {"words": "three thousand 16 aath", "tag": "unsure"}]
"Do din pehle order kiya tha, cancel kar do" -> []
