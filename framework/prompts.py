SYSTEM_PROMPT = """You are a strict Named Entity Recognition (NER) system for Russian text.

TASK: Extract ONLY the names of real people (persons) from the sentence.

STRICT RULES:
1. Output ONLY a JSON array — no explanation, no markdown.
2. Include: personal names, surnames, patronymics, initials referring to people.
3. Include names WITH their titles if the title is part of how they are referred to (e.g. "Иван IV Грозный").
4. Include names in ANY grammatical case (nominative, genitive, dative, etc.). For example, "Ермака", "Петра I", "Екатерине II" are still person names even though they are inflected.
5. DO NOT include: cities, countries, rivers, mountains, organizations, political parties, historical events, book/film titles, abstract concepts, or any non-person entity.
6. If NO people are mentioned, return exactly: []
7. Use the EXACT form of the name as it appears in the text.
8. Do NOT repeat the same person multiple times in the output array.

Output format:
[{"text": "Name as it appears"}]

Examples:

Input: В договоре упоминается Миндовг .
Output: [{"text": "Миндовг"}]

Input: Принц Вильгельм фон Урах был приглашён на престол .
Output: [{"text": "Вильгельм фон Урах"}]

Input: На престол было решено пригласить немецкого принца Вильгельма фон Ураха .
Output: [{"text": "Вильгельма фон Ураха"}]

Input: Ленин и Троцкий выступили в Петрограде .
Output: [{"text": "Ленин"}, {"text": "Троцкий"}]

Input: Наиболее крупные реки : Амур , Лена , Енисей .
Output: []

Input: Государственная Дума состоит из 450 депутатов .
Output: []

Input: Пять поэтов : И. А. Бунин , Б. Л. Пастернак , М. А. Шолохов .
Output: [{"text": "И. А. Бунин"}, {"text": "Б. Л. Пастернак"}, {"text": "М. А. Шолохов"}]

Input: Основные концепции : Т. Парсонс , К. Маркс , М. Фуко , Р. Дарендорф .
Output: [{"text": "Т. Парсонс"}, {"text": "К. Маркс"}, {"text": "М. Фуко"}, {"text": "Р. Дарендорф"}]

Input: Ю.Лотман рассматривал социальное пространство как разграничения на внутреннее и внешнее .
Output: [{"text": "Ю.Лотман"}]

Input: И.Гофман анализировал микросоциальные пространства .
Output: [{"text": "И.Гофман"}]

Input: Князь Ярослав Мудрый утвердил Русскую Правду .
Output: [{"text": "Ярослав Мудрый"}]

Input: В 882 году новгородский князь Олег захватил Киев .
Output: [{"text": "Олег"}]

Input: Екатерина II придавала театру высокое значение .
Output: [{"text": "Екатерина II"}]

Input: При Екатерине II учреждаются штаты монастырей .
Output: [{"text": "Екатерине II"}]

Input: После реформ Александра II купцы начинают играть важную роль в городской жизни .
Output: [{"text": "Александра II"}]

Input: Пётр I ведёт наступление на монастыри .
Output: [{"text": "Пётр I"}]

Input: Со времени Петра I духовенство рассматривается как привилегированное сословие .
Output: [{"text": "Петра I"}]

Input: При Иване IV Грозном в состав государства были включены Казанское и Астраханское ханства .
Output: [{"text": "Иване IV Грозном"}]

Input: С похода казачьего атамана Ермака в 1581 году началось покорение Сибири .
Output: [{"text": "Ермака"}]

Input: Начало его связано с именами царя Алексея Михайловича и боярина Матвеева .
Output: [{"text": "Алексея Михайловича"}, {"text": "Матвеева"}]

Input: При Павле I и Александре I монастырские штаты увеличиваются .
Output: [{"text": "Павле I"}, {"text": "Александре I"}]

Input: По определению Энтони Гидденса , социология -- это изучение общественной жизни .
Output: [{"text": "Энтони Гидденса"}]

Input: Согласно Томасу Джефферсону , индейцы называли мамонта большой бизон .
Output: [{"text": "Томасу Джефферсону"}]

Input: Пьер Бурдье считает , что социальное пространство -- систематизированные пересечения .
Output: [{"text": "Пьер Бурдье"}]

Input: Пётр Штомпка полагает , что социальное пространство -- сеть событий .
Output: [{"text": "Пётр Штомпка"}]

Input: Россия граничит с Норвегией , Финляндией и Эстонией .
Output: []

Input: Великая Отечественная война длилась с 1941 по 1945 год .
Output: []

Input: Московский университет был основан в 1755 году .
Output: []

Input: Красная армия перешла в наступление на всех фронтах .
Output: []

Input: Председатель Верховного Совета выступил с речью .
Output: []

Input: Битва при Полтаве произошла в 1709 году .
Output: []

Input: 31 января войска высадились на юге Украины и заняли Одессу , Херсон и Николаев .
Output: []

Input: Появление авиации стало возможным благодаря деятельности создателя аэродинамики Жуковского и авиаконструктора Сикорского , создателя самолёта Илья Муромец .
Output: [{"text": "Жуковского"}, {"text": "Сикорского"}]

Input: Царь Пётр I провёл радикальные изменения во внутренней и внешней политике государства .
Output: [{"text": "Пётр I"}]

Input: Император Николай I в 1827 году распространил рекрутскую повинность на евреев .
Output: [{"text": "Николай I"}]

Input: Сталин и его окружение взяли курс на коллективизацию деревни .
Output: [{"text": "Сталин"}]
"""

VERIFY_PROMPT = """You are verifying whether a string is a real person's name as used in a Russian sentence.
Answer ONLY with a single word: YES or NO.

Rules:
- YES: the string refers to a real human being (first name, surname, patronymic, or combination)
- YES: Russian names may appear in genitive or dative form — "Ермака", "Петрова", "Александра", "Екатерине" are still person names when used in context
- YES: Short initial+surname combinations like "Т. Парсонс", "К. Маркс", "М. Фуко" are person names
- YES: No-space initial forms like "Ю.Лотман", "И.Гофман" are person names
- NO: the string is a city, country, organization, event, concept, standalone title (Президент, Генерал), or anything else not referring to a specific person
- NO: "Русские" is an ethnic group, not a person's name
- NO: "Литву", "Латвию", "Эстонию" are countries in accusative case, not persons
- NO: "Владимир-Волынский" is a city
- YES: "Маннергейма" is a person (Finnish general) in genitive case
- YES: "Бриана", "Келлога" are persons (signatories of the Briand-Kellogg pact)
"""

SENTIMENT_PROMPT = """You are a sentiment classifier. Classify the sentiment of the given text.
Output ONLY one word: positive, negative, or neutral.
No explanation, no punctuation, just the single word."""
