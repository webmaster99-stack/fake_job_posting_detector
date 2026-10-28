// Form state for a posting, and the choices offered for its categorical fields.
// The categories are EMSCAD's values (the training data); free text is still accepted.

import type { Posting } from "./api";

export type TextField = Exclude<keyof Posting, "telecommuting" | "has_company_logo" | "has_questions">;
export type FlagField = "telecommuting" | "has_company_logo" | "has_questions";

export type PostingForm = Record<TextField, string> & Record<FlagField, boolean>;

export const EMPTY_FORM: PostingForm = {
  title: "",
  company_profile: "",
  description: "",
  requirements: "",
  benefits: "",
  location: "",
  department: "",
  salary_range: "",
  employment_type: "",
  required_experience: "",
  required_education: "",
  industry: "",
  function: "",
  telecommuting: false,
  has_company_logo: false,
  has_questions: false,
};

/** The fields whose text the model reads as words (and where terms are highlighted). */
export const LONG_TEXT_FIELDS = [
  "company_profile",
  "description",
  "requirements",
  "benefits",
] as const satisfies readonly TextField[];

/** Request body: blank fields are left out, because a missing field is itself a signal. */
export function toPosting(form: PostingForm): Posting {
  const posting: Partial<Posting> = {
    telecommuting: form.telecommuting,
    has_company_logo: form.has_company_logo,
    has_questions: form.has_questions,
  };
  for (const [key, value] of Object.entries(form)) {
    if (typeof value === "string" && value.trim()) {
      (posting as Record<string, unknown>)[key] = value.trim();
    }
  }
  return posting as Posting;
}

export const CHOICES = {
  employment_type: ["Full-time", "Part-time", "Contract", "Temporary", "Other"],
  required_experience: [
    "Internship",
    "Entry level",
    "Associate",
    "Mid-Senior level",
    "Director",
    "Executive",
    "Not Applicable",
  ],
  required_education: [
    "Some High School Coursework",
    "High School or equivalent",
    "Vocational - HS Diploma",
    "Vocational",
    "Vocational - Degree",
    "Some College Coursework Completed",
    "Associate Degree",
    "Certification",
    "Bachelor's Degree",
    "Master's Degree",
    "Doctorate",
    "Professional",
    "Unspecified",
  ],
  function: [
    "Accounting/Auditing",
    "Administrative",
    "Advertising",
    "Art/Creative",
    "Business Analyst",
    "Business Development",
    "Consulting",
    "Customer Service",
    "Data Analyst",
    "Design",
    "Distribution",
    "Education",
    "Engineering",
    "Finance",
    "Financial Analyst",
    "General Business",
    "Health Care Provider",
    "Human Resources",
    "Information Technology",
    "Legal",
    "Management",
    "Manufacturing",
    "Marketing",
    "Product Management",
    "Production",
    "Project Management",
    "Public Relations",
    "Purchasing",
    "Quality Assurance",
    "Research",
    "Sales",
    "Science",
    "Strategy/Planning",
    "Supply Chain",
    "Training",
    "Writing/Editing",
    "Other",
  ],
  industry: [
    "Computer Software",
    "Construction",
    "Consumer Services",
    "Design",
    "E-Learning",
    "Education Management",
    "Financial Services",
    "Hospital & Health Care",
    "Information Technology and Services",
    "Internet",
    "Marketing and Advertising",
    "Oil & Energy",
    "Real Estate",
    "Retail",
    "Telecommunications",
  ],
} as const;

// Invented postings for trying the form out.
export const EXAMPLES: { name: string; form: PostingForm }[] = [
  {
    name: "Suspicious example",
    form: {
      ...EMPTY_FORM,
      title: "Data Entry Clerk – Work From Home",
      description:
        "Earn $500–$900 weekly from the comfort of your home! No experience needed, flexible " +
        "hours, start immediately. We provide all training. To apply, send your full name, " +
        "phone number and bank details for direct deposit setup to hiring.dept@quickpay-jobs.net.",
      employment_type: "Part-time",
      telecommuting: true,
    },
  },
  {
    name: "Ordinary example",
    form: {
      ...EMPTY_FORM,
      title: "Junior Backend Developer",
      company_profile:
        "We build scheduling software for independent clinics. Our team of 40 works from our " +
        "office in Berlin and has shipped to more than 2,000 practices since 2016.",
      description:
        "You will work on our Python services and PostgreSQL database with a senior mentor, " +
        "reviewing code together, writing tests and taking part in planning each sprint.",
      requirements:
        "Some experience with Python or another backend language. Familiarity with SQL and git. " +
        "Good written English.",
      benefits: "30 days of paid leave, a learning budget, and a public transport pass.",
      location: "DE, BE, Berlin",
      department: "Engineering",
      employment_type: "Full-time",
      required_experience: "Entry level",
      required_education: "Bachelor's Degree",
      industry: "Computer Software",
      function: "Engineering",
      has_company_logo: true,
      has_questions: true,
    },
  },
];
