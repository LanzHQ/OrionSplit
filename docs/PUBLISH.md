# Как опубликовать релиз на GitHub

Всё готово к загрузке. Ниже — по шагам, через браузер.

## Что куда пойдёт

| Файл | Куда |
|---|---|
| `release\OrionSplit-2.6-win64.7z` | вложение релиза |
| `release\melband_roformer_instvox_duality_v2.safetensors` | вложение релиза |
| `docs\banner.png` | в репозиторий + в описание релиза |
| `docs\README_public.md` | станет `README.md` репозитория |
| `docs\RELEASE_NOTES.md` | текст описания релиза |
| исходники (`app.py`, `engine\`, `ui\`, `tools\`…) | в репозиторий |

---

## Шаг 1. Создать репозиторий

1. Открой https://github.com/new
2. **Repository name:** `OrionSplit`
3. **Description:** `Удаление диалогов из аудио и видео. Mel-Band RoFormer, GPU, портативная сборка под Windows.`
4. Выбери **Public**
5. Ничего не отмечай в «Initialize this repository» — файлы зальём сами
6. Нажми **Create repository**

## Шаг 2. Залить исходники

Проще всего одной командой из папки проекта. Подставь свой ник вместо
`ТВОЙ_НИК`:

```bash
cd /e/InstVoxRender && git init -b main && cp docs/README_public.md README.md && git add -A && git commit -m "OrionSplit v2.6" && git remote add origin https://github.com/ТВОЙ_НИК/OrionSplit.git && git push -u origin main
```

При первом `git push` откроется окно входа в GitHub — авторизуйся в нём.

`.gitignore` уже настроен: сборка, модель, venv и ffmpeg в репозиторий
не попадут — только исходники, документация и баннер.

> Если возиться с git не хочется: на странице пустого репозитория нажми
> **uploading an existing file** и перетащи туда `app.py`, папки `engine`,
> `ui`, `tools`, `docs`, файлы `requirements.txt`, `build_gpu.bat`,
> `build_cpu.bat`, `.gitignore` и `README.md` (переименованную копию
> `docs\README_public.md`).

## Шаг 3. Создать релиз

1. Зайди в репозиторий → справа **Releases** → **Create a new release**
2. **Choose a tag** → впиши `v2.6` → **Create new tag: v2.6 on publish**
3. **Release title:** `OrionSplit v2.6`
4. В поле описания вставь текст из `docs\RELEASE_NOTES.md`
5. **Перетащи `docs\banner.png` прямо в поле описания** — GitHub загрузит
   картинку и сам подставит ссылку. Перетащи в самое начало текста.
6. Перетащи в блок **Attach binaries** оба файла из папки `release\`:
   - `OrionSplit-2.6-win64.7z`
   - `melband_roformer_instvox_duality_v2.safetensors`
   Загрузка нескольких гигабайт займёт время — не закрывай вкладку.
7. Галочку **Set as the latest release** оставь включённой
8. **Publish release**

## Шаг 4. Проверить

- Ссылка `Скачать` в README ведёт на `/releases/latest` — она заработает
  сразу после публикации релиза.
- Открой страницу релиза в режиме инкогнито и убедись, что оба файла
  скачиваются без входа в аккаунт.

---

## Перед публикацией стоит проверить

- **Лицензия модели.** Модель `Mel-Band RoFormer InstVox Duality v2`
  чужая (автор pcunwa, HuggingFace). Загляни на её страницу и убедись,
  что перераспространение разрешено. Если условия непонятны — безопаснее
  не выкладывать файл модели, а дать в README ссылку на HuggingFace,
  чтобы пользователь скачивал её сам.
- **FFmpeg под GPL.** В архив вложен `LICENSES.txt` с указанием лицензий
  и ссылкой на исходники FFmpeg — этого обычно достаточно, поскольку он
  вызывается как отдельная программа.
- **Антивирусы.** Программы, собранные PyInstaller, иногда ложно
  срабатывают у Windows Defender и на VirusTotal. Если пойдут жалобы —
  имеет смысл отправить сборку в Microsoft на анализ ложного
  срабатывания: https://www.microsoft.com/wdsi/filesubmission
