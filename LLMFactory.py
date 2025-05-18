import re
import logging
from typing import Any, Dict, List, Optional, Union
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage
import functools
from datetime import datetime

logger = logging.getLogger(__name__)

_language_model_registry = {
    "private_llm": {
        "model_name": "amazon.nova-pro-v1:0",
        "model_provider": "bedrock_converse"
    },
    "small_llm": {
        #"model_name": "amazon.nova-pro-v1:0",
        #"model_provider": "bedrock_converse"
        "model_name": "gpt-4o-mini",
        "model_provider": "openai"
    },
    "large_llm": {
        "model_name": "amazon.nova-pro-v1:0",
        "model_provider": "bedrock_converse"
    }
}

class LLMFactory:
    """Factory for creating and managing LLM instances."""

    @staticmethod
    def get_llm(llm_type: str) -> Any:
        """Get an LLM instance based on the model name and provider."""

        if llm_type not in _language_model_registry:
            raise ValueError(f"Invalid LLM type: {llm_type}")

        model_config = _language_model_registry[llm_type]
        return init_chat_model(model_config["model_name"], model_provider=model_config["model_provider"])

    @staticmethod
    async def generate_llm_response(
        messages: List[BaseMessage],
        model_type: str = "small_llm", #default to small_llm
        system_prompt: Optional[str] = None, # Caller now provides the full system prompt if needed
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        clean_response_content: bool = True,
        **kwargs
    ) -> AIMessage:
        """
        Generates a response from an LLM, incorporating dynamic guidance and optional cleaning.

        This method fetches the LLM, applies channel/type-specific guidance,
        allows for an additional system prompt, invokes the LLM, and optionally cleans the response.

        Args:
            model_type: Type of model to use (e.g., "small_llm", "large_llm").
            messages: List of BaseMessage objects for the conversation history.
                        The caller is responsible for ensuring messages are in the correct format.
            system_prompt: Optional overarching system prompt string. This will be prepended
                           to any dynamically generated guidance.
            temperature: Optional temperature setting for the LLM. If None, model default is used.
            max_tokens: Optional max tokens setting for the LLM. If None, model default is used.
            clean_response_content: If True, cleans the LLM response content using clean_response.
            **kwargs: Additional keyword arguments to pass to the LLM's ainvoke method.

        Returns:
            The AIMessage response from the LLM, with its content potentially cleaned.
        """
        llm_instance = LLMFactory.get_llm(model_type)

        processed_messages = [msg for msg in messages] # Create a mutable copy

        if system_prompt: # Simplified: use system_prompt directly
            existing_system_message_index = -1
            for i, msg in enumerate(processed_messages):
                if isinstance(msg, SystemMessage):
                    existing_system_message_index = i
                    break
            
            if existing_system_message_index != -1:
                # Prepend to existing system message's content if caller wishes to augment
                # Or, caller can choose to replace by ensuring only one system message is in processed_messages
                original_content = processed_messages[existing_system_message_index].content
                processed_messages[existing_system_message_index].content = f"{system_prompt}\\n\\n{original_content}"
            else:
                processed_messages.insert(0, SystemMessage(content=system_prompt))

        invoke_params = {}
        if temperature is not None:
            invoke_params['temperature'] = temperature
        if max_tokens is not None:
            invoke_params['max_tokens'] = max_tokens
        invoke_params.update(kwargs)

        logger.info(f"[LLM Request] Model: {model_type}, Temp: {invoke_params.get('temperature', 'default')}, Messages: {len(processed_messages)}")

        try:
            response = await llm_instance.ainvoke(
                processed_messages,
                    **invoke_params
                )
            
            logger.info(f"[LLM Response] Raw response content: {response.content}")

            #if clean_response_content:
            #    response.content = LLMFactory._clean_llm_response(response.content, clean_response_content)

            response_content = ""
            if isinstance(response.content, list):
                response_content = "".join([item.get('text', '') for item in response.content])
            else:
                response_content = response.content

            # Preserve the full response object, but clean its content if requested
            if clean_response_content:
                response.content = LLMFactory.clean_response(response_content) # Modify content in-place

            return response # Return the full AIMessage (or other BaseMessage) object

        except Exception as e:
            logger.error(f"[LLM Error] Model: {model_type}, Error: {str(e)}")
            # Consider if a more specific exception should be raised or if returning a specific AIMessage indicating error
            # For now, re-raising to match previous behavior.
            raise

    @staticmethod
    def get_channel_instructions(channel_type: str) -> str:
        """Get channel instructions based on the channel type."""
        if channel_type == "voice":
            channel_instructions = (
                "\n\nYou are responding over a VOICE channel. Keep these guidelines in mind:"
                "\n- Keep responses concise (1-3 sentences when possible)"
                "\n- Avoid URLs, special characters, and complex formatting"
                "\n- Use simple language suitable for speaking aloud"
                "\n- Format phone numbers with pauses (e.g., '800. 555. 1234')"
                "\n- Spell out important information that might be misheard"
                "\n- Use numbers instead of digits for clarity (e.g., 'thirty dollars' instead of '$30')"
            )
            return channel_instructions
        elif channel_type == "sms":
            channel_instructions = (
                "\n\nYou are responding over an SMS channel. Keep these guidelines in mind:"
                "\n- Keep responses short and to the point"
                "\n- Break complex information into multiple short paragraphs"
                "\n- Use plain text formatting only"
                "\n- Avoid sending very long messages"
            )
            return channel_instructions
        return ""

    @staticmethod
    def _get_channel_guidance(channel_type: str) -> str:
        """Private helper to get channel-specific guidance strings."""
        # Calls the existing public method to leverage defined instructions
        return LLMFactory.get_channel_instructions(channel_type)

    @staticmethod
    def get_date_guidance() -> str:
        """
        Private helper to get date-specific guidance strings.
        Includes a placeholder for the current date.
        """
        
        return (
            "\n\n--- Date Guidance (Current Date: {current_date_iso}) ---\n"
            "When processing dates, you MUST use the provided current date ({current_date_iso}) to interpret user input accurately:\n"
            "1. Relative Dates: If the user says 'next Tuesday', 'yesterday', or 'in two weeks', calculate the specific date based on the current date. Always confirm this calculated date with the user (e.g., 'So, that would be Tuesday, YYYY-MM-DD. Is that correct?').\n"
            "2. Incomplete Dates (e.g., MM/DD like '03/15'): Infer the year. If 'MM/DD' refers to a date that has already passed this year (based on {current_date_iso}), assume it's for the next year. If it's a future date within the current year, assume the current year. Always state your assumption and ask for confirmation (e.g., 'You mentioned March 15th. Are you referring to March 15, [inferred_year]?').\n"
            "3. Ambiguity: If a date is ambiguous (e.g., 'the 5th'), ask for the month and year.\n"
            "4. Format: When confirming or stating dates, use a clear, unambiguous format (e.g., YYYY-MM-DD or Month DD, YYYY)."
            "\n--- End Date Guidance ---"
        )

    @staticmethod
    def get_time_guidance() -> str:
        """Private helper to get time-specific guidance strings."""
        return (
            "\n\n--- Time Guidance ---\n"
            "When processing times:\n"
            "1. AM/PM: If the user states a time without AM/PM (e.g., 'at 7'), infer AM/PM based on context (e.g., 'dinner at 7' implies PM). If genuinely ambiguous (e.g., 'meeting at 7'), ask for clarification (e.g., 'Is that 7 AM or 7 PM?').\n"
            "2. Relative Slots (e.g., 'early morning', 'late afternoon', 'evening'): Ask the user to specify a more precise time or offer a suggested range. For example: 'When you say early morning, do you mean around 7-9 AM?' or 'For late afternoon, would 4-6 PM work?'. \n"
            "3. Time Zones: Be mindful of time zones if the context involves different geographical locations. If unsure, assume the user's local time zone or ask for clarification.\n"
            "4. Format: When confirming or stating times, use a clear format (e.g., HH:MM AM/PM or 24-hour format if appropriate for the context)."
            "\n--- End Time Guidance ---"
        )

    @staticmethod
    def _get_custom_field_guidance(field_identifier: str) -> str:
        """Private helper to get custom field specific guidance strings."""
        # This can be expanded with a dictionary lookup for field-specific rules
        return (
            f"\n\n--- Custom Field: {field_identifier} ---\n"
            f"When handling the field '{field_identifier}', ensure you collect all necessary details. "
            f"If specific validation rules or formats apply to '{field_identifier}', adhere to them. "
            f"Confirm the provided information for '{field_identifier}' with the user if it seems unusual or critical."
            f"\n--- End Custom Field: {field_identifier} ---"
        )

    @staticmethod
    def _clean_llm_response(response: any, clean_response_content: bool = True) -> str:
        if isinstance(response, str):
            # Content is already a string, use it directly
            final_text_response = response
        elif isinstance(response, list):
            # Content is a list, likely [{'type': 'text', 'text': '...'}]
            text_parts = []
            for item in response:
                # Handle potential non-dict items or items without 'type'/'text'
                if isinstance(item, dict) and item.get('type') == 'text':
                        text_parts.append(item.get('text', '')) # Extract text, default to empty if key missing
            final_text_response = "".join(text_parts) # Join parts (usually just one)
        else:
            # Unexpected content type, try basic string conversion as fallback
            logger.warning(f"Unexpected AIMessage content type: {type(response)}. Attempting string conversion.")
            final_text_response = str(response)

        if clean_response_content:
            return LLMFactory.clean_response(final_text_response)
        else:
            return final_text_response

    @staticmethod
    def clean_response(response: Union[str, List[str]]) -> str:

        """Some LLM's include reasoning within their response
        We don't want end users seeing this, so we clean it up.

        Args:
            response (str): string response from Langgraph/LLM.
            
        Returns:
            str: The cleaned response with thinking tags removed
        """

        src_text = ""
        # Handle non-string inputs
        if isinstance(response, list):
            msg = response[0]
            src_text = msg.get("text", "")
        else:
            src_text = response


                    
        # Define regex pattern
        thinking_pattern = r'<thinking>.*?</thinking>'

        cleaned_content = re.sub(thinking_pattern, "", src_text, flags=re.DOTALL)

        return cleaned_content
