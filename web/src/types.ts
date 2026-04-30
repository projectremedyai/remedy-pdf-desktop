export interface TextDeltaEvent {
  type: "text_delta";
  content: string;
}

export interface ToolCallEvent {
  type: "tool_call";
  id: string;
  name: string;
  args: Record<string, unknown>;
}

export interface ToolResultEvent {
  type: "tool_result";
  id: string;
  name: string;
  result: unknown;
}

export interface ErrorEvent {
  type: "error";
  message: string;
}

export interface DoneEvent {
  type: "done";
}

export type ServerEvent =
  | TextDeltaEvent
  | ToolCallEvent
  | ToolResultEvent
  | ErrorEvent
  | DoneEvent;

export interface UserMessage {
  type: "message";
  content: string;
  campus: string;
  history: Array<{ role: string; content: string }>;
}

export interface ToolCall {
  id: string;
  name: string;
  args: Record<string, unknown>;
  result?: unknown;
  status: "pending" | "done";
}

export interface Message {
  role: "user" | "assistant";
  content: string;
  toolCalls?: ToolCall[];
}

export type ConnectionState = "connecting" | "connected" | "disconnected";
