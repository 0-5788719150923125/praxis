/**
 * Praxis Web - a streaming assistant turn
 *
 * The model decodes inside the training loop, one request per step, so a turn
 * takes as long as it takes whether or not anyone is watching. Streaming does
 * not change that - it changes a 30-60s blank wait into text appearing, which
 * is the whole difference between "thinking" and "broken".
 *
 * The turn's final content always comes from the POST's own answer, never from
 * the accumulated deltas. They should agree (server-side, both go through the
 * same reply extractor), and settling on the authoritative one anyway is what
 * makes a dropped frame or a socket that was down cost nothing.
 */

import { state } from './state.js';
import { render, renderStreamingMessages } from './render.js';

// Shown when the request produced nothing AND nothing was streamed. A turn that
// DID stream keeps its text instead - see `settle`. (A genuinely empty model
// turn is not this: the server substitutes its own placeholder for that, so an
// empty body here means the request timed out or failed.)
const NO_ANSWER = 'Error: No response';

/**
 * Begin an assistant turn that fills in as the reply arrives.
 *
 * Nothing is pushed into `state.messages` until the first delta lands, so the
 * conversation sent to the API never contains the turn being generated, and a
 * request that streams nothing behaves exactly as it did before.
 *
 * @returns {{onDelta: function, onReset: function, settle: function, fail: function}}
 */
export function streamingTurn() {
    let turn = null;
    let pendingFrame = null;

    /** Coalesce repaints to one per frame, and repaint only the conversation.
     *  A byte-level model emits a delta per byte; walking the whole app for
     *  each one is waste, and `renderStreamingMessages` patches the text into
     *  the nodes already on the page rather than rebuilding the list. */
    const scheduleRender = () => {
        if (pendingFrame !== null) return;
        pendingFrame = requestAnimationFrame(() => {
            pendingFrame = null;
            renderStreamingMessages();
        });
    };

    const ensureTurn = () => {
        if (!turn) {
            turn = { role: 'assistant', content: '', streaming: true };
            state.messages.push(turn);
            // Text is arriving, so the thinking dots have done their job.
            state.isThinking = false;
        }
        return turn;
    };

    return {
        onDelta(text) {
            ensureTurn().content += text;
            scheduleRender();
        },

        /** The runtime spliced a tool result and moved the turn anchor past it:
         *  everything shown so far has stopped being part of the answer. */
        onReset() {
            if (turn) {
                turn.content = '';
                scheduleRender();
            }
        },

        /**
         * Replace the streamed text with the authoritative reply.
         *
         * Authoritative WHEN IT HAS ONE. An empty answer is not a better
         * account of the turn than the text the reader already watched arrive:
         * `POST /messages/` returns `""` when the client's own 60s patience ran
         * out mid-reply (256 bytes of uncached byte-level decoding routinely
         * takes longer) or when the route swallowed an error, and replacing a
         * long visible reply with "Error: No response" at the very end is the
         * one outcome worse than either.
         *
         * The stream is closed by then, so what is on screen is the whole of
         * what the model produced before the request was abandoned. Keep it,
         * and say it was cut short.
         */
        settle(content) {
            // The fallback lives HERE, not at the call sites: they used to pass
            // `response.response || 'Error: No response'`, so `settle` never saw
            // an empty answer and could not tell "the model said nothing" from
            // "we already showed the reader a page of text".
            content = content || '';
            if (turn && !content && turn.content) {
                return finishTurn(
                    turn.content,
                    'cut short - the request timed out while this was still being written'
                );
            }
            const answer = content || NO_ANSWER;
            if (turn) return finishTurn(answer);
            state.messages.push({ role: 'assistant', content: answer });
        },

        /**
         * The request failed outright.
         *
         * Same rule: a partial reply the reader watched arrive beats an error
         * bubble that erases it, so the error becomes a footnote on the text
         * rather than a replacement for it. With nothing streamed there is
         * nothing to keep, and the error is the whole message.
         */
        fail(message) {
            if (turn && turn.content) return finishTurn(turn.content, message);
            discardTurn();
            state.messages.push({ role: 'assistant', content: message });
        }
    };

    function finishTurn(content, caption) {
        turn.content = content;
        delete turn.streaming;
        if (caption) turn.caption = caption;
        turn = null;
    }

    function discardTurn() {
        if (!turn) return;
        const index = state.messages.indexOf(turn);
        if (index !== -1) state.messages.splice(index, 1);
        turn = null;
    }
}
