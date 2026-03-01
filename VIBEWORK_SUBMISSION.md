# VibeWork – Hackathon Submission

## Inspiration

We wanted an AI that doesn’t just follow fixed scripts—it should **see your screen**, understand what you’re doing, and help with real tasks: opening apps, navigating the UI, or even playing chess when you have a board open. We were inspired by the idea of a single agent that can switch between “assistant” and “player” based on context, and that **learns from your corrections** instead of repeating the same mistakes.

## What it does

**VibeWork** is a desktop AI agent you control with natural language. You type a task (e.g. *“play chess”* or *“open Notepad”*), and it:

- **Takes a screenshot** and sends it to a vision model (Gemini).
- **Routes your intent**: if it sees a chess board, it starts the chess bot (YOLO piece detection + Stockfish); otherwise it controls your screen—clicking, typing, using the keyboard.
- **Runs in a loop**: after each action it takes another screenshot and asks “what’s next?” until the task is done or you stop it.
- **Learns from feedback**: when you correct it (e.g. “my pieces are at the bottom—that means I’m black”), it stores that and injects it into future prompts so the model follows your rules next time.

So in one app you get: **vision-based task routing**, **screen control**, **chess autoplay**, and **feedback-driven improvement**.

## How we built it

- **Frontend & orchestration:** Python + Tkinter GUI; one text box for the task and a log for actions and model reasoning.
- **Vision & routing:** Google Gemini (vision) for screenshot understanding and JSON action output (e.g. `MOVE_TO`, `CLICK`, `start_chess`). We use the API’s **system instruction** so user feedback is treated as rules, not just context.
- **Screen control:** PyAutoGUI for mouse/keyboard; PyWinAuto on Windows for UI automation (click by element name when available).
- **Chess:** YOLOv8 for piece detection on the board, Python-Chess + Stockfish for move generation; the agent detects “playing as” from which pieces are at the bottom of the screen.
- **Learning:** User corrections (and optional self-feedback) are saved in `prompt_history.json` and turned into a “LEARNING FROM USER FEEDBACK” block that’s added to the system prompt so the model applies those corrections on later runs.

## Challenges we ran into

- **Model sometimes replied with only `<thinking>` and no JSON** → We added a single automatic retry: if there’s no action JSON, we send a short follow-up (“Reply with exactly one JSON object”) and parse that before giving up, so the loop doesn’t stop unnecessarily.
- **Model ignoring user feedback** → We moved from concatenating the system prompt with the user message to using Gemini’s **system_instruction** parameter so feedback is in the “instructions” channel; we also made the feedback block explicitly say “you MUST apply these corrections.”
- **Chess “playing as” wrong** → Users kept saying “my pieces are at the bottom.” We tightened the router prompt and made feedback include a concrete rule (e.g. “bottom two rows = player color”) and reinforced it in the learning block.
- **High-res screenshots and context limits** → We resize screenshots for the model and only send the last 2 prior screenshots with captions to balance context vs. token use.

## Accomplishments that we're proud of

- **Single agent, multiple modes:** One entry point (“type a task”) that can start a chess game or drive your desktop, with no mode switch in the UI.
- **Feedback that actually sticks:** Stored feedback is injected as system instructions so the model consistently applies user corrections (e.g. chess color, “don’t click if it’s already open”).
- **Robustness:** Retry when the model forgets to output JSON, and clearer prompts so the agent keeps going instead of stopping on the first bad reply.
- **End-to-end on Windows:** Screenshot → Gemini → parse → execute (mouse/keyboard/UI automation) → repeat, with chess as a first-class flow when a board is visible.

## What we learned

- **System vs. user prompt matters:** Putting instructions (and feedback) in the system instruction made the model follow them much more reliably than stuffing everything into the user message.
- **One retry can fix a lot:** A single, targeted follow-up (“output JSON only”) often recovers from thinking-only or malformed responses without complex parsing.
- **User wording is gold:** Short, concrete feedback (“bottom two rows = my color”) works better than long explanations when we inject it back into the prompt.
- **Vision + automation is hard but doable:** Coordinating screenshot size, cursor overlay, and action space (one action per turn, move-then-click) was critical for the agent to behave predictably.

## What's next for VibeWork

- **More feedback triggers:** Optional “lesson” after success (e.g. “that worked but was slow—here’s a faster way”) and better grouping of feedback by task type (chess vs. general UI).
- **Smarter screenshot history:** Optional summarization or selection of which prior screens to send so we can use more history without blowing the context window.
- **Multi-monitor and DPI:** Correct coordinate mapping and scaling across monitors and different DPIs.
- **Voice in/out:** Speak the task and hear status so you can use VibeWork hands-free.
- **Presets and macros:** Save common flows (e.g. “open Slack and focus channel X”) and reuse them with one click or phrase.
