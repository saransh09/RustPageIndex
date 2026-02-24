//! LLM prompts for PageIndex operations.
//!
//! These prompts are ported from the original Python PageIndex implementation.

/// Collection of prompts used for tree generation and search.
pub struct Prompts;

impl Prompts {
    /// Prompt to detect if a page contains a table of contents.
    pub fn toc_detector() -> &'static str {
        r#"Your job is to detect if there is a table of content provided in the given text.

Given text: {content}

return the following JSON format:
{
    "thinking": <why do you think there is a table of content in the given text>
    "toc_detected": "<yes or no>",
}

Directly return the final JSON structure. Do not output anything else.
Please note: abstract, summary, notation list, figure list, table list, etc. are not table of contents."#
    }

    /// Prompt to transform raw TOC into structured JSON.
    pub fn toc_transformer() -> &'static str {
        r#"You are given a table of contents, You job is to transform the whole table of content into a JSON format included table_of_contents.

structure is the numeric system which represents the index of the hierarchy section in the table of contents. For example, the first section has structure index 1, the first subsection has structure index 1.1, the second subsection has structure index 1.2, etc.

The response should be in the following JSON format: 
{
"table_of_contents": [
    {
        "structure": <structure index, "x.x.x" or None> (string),
        "title": <title of the section>,
        "page": <page number or None>,
    },
    ...
    ],
}
You should transform the full table of contents in one go.
Directly return the final JSON structure, do not output anything else."#
    }

    /// Prompt to generate initial tree structure from document text.
    pub fn generate_toc_init() -> &'static str {
        r#"You are an expert in extracting hierarchical tree structure, your task is to generate the tree structure of the document.

The structure variable is the numeric system which represents the index of the hierarchy section in the table of contents. For example, the first section has structure index 1, the first subsection has structure index 1.1, the second subsection has structure index 1.2, etc.

For the title, you need to extract the original title from the text, only fix the space inconsistency.

The provided text contains tags like <physical_index_X> and <physical_index_X> to indicate the start and end of page X.

For the physical_index, you need to extract the physical index of the start of the section from the text. Keep the <physical_index_X> format.

The response should be in the following format: 
    [
        {
            "structure": <structure index, "x.x.x"> (string),
            "title": <title of the section, keep the original title>,
            "physical_index": "<physical_index_X> (keep the format)"
        },
        
    ],


Directly return the final JSON structure. Do not output anything else."#
    }

    /// Prompt to continue generating tree structure from additional document parts.
    pub fn generate_toc_continue() -> &'static str {
        r#"You are an expert in extracting hierarchical tree structure.
You are given a tree structure of the previous part and the text of the current part.
Your task is to continue the tree structure from the previous part to include the current part.

The structure variable is the numeric system which represents the index of the hierarchy section in the table of contents. For example, the first section has structure index 1, the first subsection has structure index 1.1, the second subsection has structure index 1.2, etc.

For the title, you need to extract the original title from the text, only fix the space inconsistency.

The provided text contains tags like <physical_index_X> and <physical_index_X> to indicate the start and end of page X.

For the physical_index, you need to extract the physical index of the start of the section from the text. Keep the <physical_index_X> format.

The response should be in the following format. 
    [
        {
            "structure": <structure index, "x.x.x"> (string),
            "title": <title of the section, keep the original title>,
            "physical_index": "<physical_index_X> (keep the format)"
        },
        ...
    ]    

Directly return the additional part of the final JSON structure. Do not output anything else."#
    }

    /// Prompt to check if a section title appears on a specific page.
    pub fn check_title_appearance() -> &'static str {
        r#"Your job is to check if the given section appears or starts in the given page_text.

Note: do fuzzy matching, ignore any space inconsistency in the page_text.

The given section title is {title}.
The given page_text is {page_text}.

Reply format:
{
    "thinking": <why do you think the section appears or starts in the page_text>
    "answer": "yes or no" (yes if the section appears or starts in the page_text, no otherwise)
}
Directly return the final JSON structure. Do not output anything else."#
    }

    /// Prompt to fix incorrect TOC item page index.
    pub fn single_toc_item_index_fixer() -> &'static str {
        r#"You are given a section title and several pages of a document, your job is to find the physical index of the start page of the section in the partial document.

The provided pages contains tags like <physical_index_X> and <physical_index_X> to indicate the physical location of the page X.

Reply in a JSON format:
{
    "thinking": <explain which page, started and closed by <physical_index_X>, contains the start of this section>,
    "physical_index": "<physical_index_X>" (keep the format)
}
Directly return the final JSON structure. Do not output anything else."#
    }

    /// Prompt for tree-based search/reasoning over the document structure.
    pub fn tree_search() -> &'static str {
        r#"You are an expert at navigating hierarchical document structures to find relevant information.

You are given:
1. A query/question from the user
2. A hierarchical tree structure of a document with sections and page indices

Your task is to analyze the tree structure and identify which sections are most likely to contain information relevant to the query.

Tree structure:
{tree_structure}

User query: {query}

Reply in JSON format:
{
    "thinking": <explain your reasoning about which sections are relevant and why>,
    "relevant_sections": [
        {
            "title": <section title>,
            "start_index": <page number where section starts>,
            "end_index": <page number where section ends>,
            "relevance": <"high", "medium", or "low">,
            "reason": <why this section is relevant to the query>
        },
        ...
    ]
}

Order sections by relevance (most relevant first).
Directly return the final JSON structure. Do not output anything else."#
    }

    /// Prompt to detect if page index numbers are given in TOC.
    pub fn detect_page_index() -> &'static str {
        r#"You will be given a table of contents.

Your job is to detect if there are page numbers/indices given within the table of contents.

Given text: {toc_content}

Reply format:
{
    "thinking": <why do you think there are page numbers/indices given within the table of contents>
    "page_index_given_in_toc": "<yes or no>"
}
Directly return the final JSON structure. Do not output anything else."#
    }

    /// Prompt to add page numbers to TOC items by analyzing document content.
    pub fn add_page_number_to_toc() -> &'static str {
        r#"You are given an JSON structure of a document and a partial part of the document. Your task is to check if the title that is described in the structure is started in the partial given document.

The provided text contains tags like <physical_index_X> and <physical_index_X> to indicate the physical location of the page X. 

If the full target section starts in the partial given document, insert the given JSON structure with the "start": "yes", and "start_index": "<physical_index_X>".

If the full target section does not start in the partial given document, insert "start": "no",  "start_index": None.

The response should be in the following format. 
    [
        {
            "structure": <structure index, "x.x.x" or None> (string),
            "title": <title of the section>,
            "start": "<yes or no>",
            "physical_index": "<physical_index_X> (keep the format)" or None
        },
        ...
    ]    
The given structure contains the result of the previous part, you need to fill the result of the current part, do not change the previous result.
Directly return the final JSON structure. Do not output anything else."#
    }

    /// Prompt to generate a summary for a document section.
    pub fn generate_node_summary() -> &'static str {
        r#"You are given a section from a document. Generate a concise summary (2-3 sentences) describing the main topics and key information covered in this section.

Section Title: {title}

Section Content:
{content}

Provide ONLY the summary text, nothing else. Be specific about what information this section contains that would help someone searching for relevant content."#
    }

    /// System prompt for general document analysis.
    pub fn system_document_analyzer() -> &'static str {
        "You are an expert document analyzer. You help extract structure, navigate content, and answer questions about documents. Always respond with valid JSON when requested."
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prompts_are_not_empty() {
        assert!(!Prompts::toc_detector().is_empty());
        assert!(!Prompts::toc_transformer().is_empty());
        assert!(!Prompts::generate_toc_init().is_empty());
        assert!(!Prompts::generate_toc_continue().is_empty());
        assert!(!Prompts::check_title_appearance().is_empty());
        assert!(!Prompts::tree_search().is_empty());
        assert!(!Prompts::generate_node_summary().is_empty());
    }
}
