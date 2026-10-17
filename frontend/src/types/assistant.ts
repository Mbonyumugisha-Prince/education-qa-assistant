export type Subject =
  | "mathematics"
  | "science"
  | "python"
  | "java"
  | "technology"
  | "history"
  | "geography";

export type AnswerResponse = {
  request_id: string;
  subject: Subject;
  answer: string;
  response_type: "greeting" | "generated" ;
  duration_seconds: number;
};

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  subject: Subject;
  passage?: string | undefined;
  duration?: number;
};