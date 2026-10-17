import type { Subject } from "@/types/assistant";

export const subjects: { id: Subject; label: string }[] = [
  { id: "mathematics", label: "Mathematics" },
  { id: "science", label: "Science" },
  { id: "python", label: "Python" },
  { id: "java", label: "Java" },
  { id: "technology", label: "Technology" },
  { id: "history", label: "History" },
  { id: "geography", label: "Geography" },
];

export function subjectLabel(subject: Subject): string {
  return subjects.find((item) => item.id === subject)?.label ?? subject;
}