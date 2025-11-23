// Package openaiclient provides an OpenAI API client wrapper with resilience patterns.
//
// The package integrates with jp-go-resilience for retry and circuit breaker
// functionality, enabling robust handling of transient failures and rate limiting.
//
// Basic usage:
//
//	client := openaiclient.New("your-api-key")
//
//	resp, err := client.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
//	    Model: openai.GPT4,
//	    Messages: []openai.ChatCompletionMessage{
//	        {Role: openai.ChatMessageRoleUser, Content: "Hello!"},
//	    },
//	})
//
// With resilience wrapper:
//
//	baseClient := openaiclient.New("your-api-key")
//	resilientClient := openaiclient.NewResilientClient(baseClient, retryWrapper, cbWrapper)
package openaiclient

import (
	"context"
	"errors"

	resilience "github.com/JohnPlummer/jp-go-resilience"
	"github.com/sashabaranov/go-openai"
)

// Client defines the interface for OpenAI API operations.
// This abstraction enables proper mocking and testing without real API calls.
type Client interface {
	// CreateChatCompletion sends a chat completion request to OpenAI API.
	CreateChatCompletion(ctx context.Context, req openai.ChatCompletionRequest) (openai.ChatCompletionResponse, error)
}

// clientWrapper wraps the concrete OpenAI client to implement our interface.
// This pattern isolates the external library dependency and enables testing.
type clientWrapper struct {
	client *openai.Client
}

// New creates a new OpenAI client wrapper with the given API key.
func New(apiKey string) Client {
	return &clientWrapper{
		client: openai.NewClient(apiKey),
	}
}

// CreateChatCompletion implements the Client interface.
func (w *clientWrapper) CreateChatCompletion(ctx context.Context, req openai.ChatCompletionRequest) (openai.ChatCompletionResponse, error) {
	return w.client.CreateChatCompletion(ctx, req)
}

// ResilientAdapter adapts the Client interface to resilience.ResilientClient.
// This allows the generic jp-go-resilience package to wrap OpenAI-specific clients
// with retry and circuit breaker functionality.
//
// The adapter implements resilience.ResilientClient[openai.ChatCompletionRequest, openai.ChatCompletionResponse]
// by delegating to a Client (which may be the base client or a mock).
type ResilientAdapter struct {
	client Client
}

// NewResilientAdapter creates a new adapter wrapping a Client.
func NewResilientAdapter(client Client) resilience.ResilientClient[openai.ChatCompletionRequest, openai.ChatCompletionResponse] {
	return &ResilientAdapter{
		client: client,
	}
}

// Execute implements resilience.ResilientClient interface by delegating to the wrapped Client.
func (a *ResilientAdapter) Execute(ctx context.Context, req openai.ChatCompletionRequest) (openai.ChatCompletionResponse, error) {
	return a.client.CreateChatCompletion(ctx, req)
}

// ResilientClientAdapter adapts a resilience.ResilientClient back to Client interface.
// This allows the resilience-wrapped client to be used anywhere a Client is expected,
// maintaining compatibility with existing code.
//
// The adapter implements Client by delegating to a
// resilience.ResilientClient[openai.ChatCompletionRequest, openai.ChatCompletionResponse].
type ResilientClientAdapter struct {
	resilientClient resilience.ResilientClient[openai.ChatCompletionRequest, openai.ChatCompletionResponse]
}

// NewResilientClientAdapter creates a new adapter wrapping a resilience.ResilientClient.
func NewResilientClientAdapter(
	resilientClient resilience.ResilientClient[openai.ChatCompletionRequest, openai.ChatCompletionResponse],
) Client {
	return &ResilientClientAdapter{
		resilientClient: resilientClient,
	}
}

// CreateChatCompletion implements Client interface by delegating to the wrapped ResilientClient.
func (a *ResilientClientAdapter) CreateChatCompletion(ctx context.Context, req openai.ChatCompletionRequest) (openai.ChatCompletionResponse, error) {
	return a.resilientClient.Execute(ctx, req)
}

// ErrorClassifier implements error classification for OpenAI API errors.
// It adapts OpenAI-specific error types to work with the jp-go-resilience package,
// providing proper retry and circuit breaker behavior for OpenAI API responses.
type ErrorClassifier struct {
	baseClassifier resilience.HTTPStatusClassifier
}

// NewErrorClassifier creates a new OpenAI error classifier.
func NewErrorClassifier() *ErrorClassifier {
	return &ErrorClassifier{
		baseClassifier: *resilience.NewHTTPStatusClassifier(),
	}
}

// ExtractStatusCode extracts the HTTP status code from an OpenAI API error.
// Returns 0 if the error is not an OpenAI API error.
func ExtractStatusCode(err error) int {
	var apiErr *openai.APIError
	if errors.As(err, &apiErr) {
		return apiErr.HTTPStatusCode
	}
	return 0
}

// IsRetryable implements resilience.ErrorClassifier for OpenAI errors.
// It delegates to the base HTTP classifier after wrapping the error with status code information.
//
// Retryable errors (429, 500-504) will trigger retry behavior.
// Non-retryable errors (400, 401, 403, 404) will fail immediately.
func (c *ErrorClassifier) IsRetryable(err error) bool {
	if err == nil {
		return false
	}

	// Extract status code from OpenAI error
	statusCode := ExtractStatusCode(err)
	if statusCode != 0 {
		// Wrap error with status code for base classifier
		wrappedErr := resilience.NewStatusCodeError(statusCode, err)
		return c.baseClassifier.IsRetryable(wrappedErr)
	}

	// Fall back to base classifier for other errors
	return c.baseClassifier.IsRetryable(err)
}

// ShouldTripCircuit implements resilience.CircuitBreakerErrorClassifier for OpenAI errors.
// It delegates to the base HTTP classifier after wrapping the error with status code information.
//
// Circuit-tripping errors include auth issues (401, 403) and server errors (500-504).
// Rate limiting (429) does not trip the circuit as it is a transient condition.
func (c *ErrorClassifier) ShouldTripCircuit(err error) bool {
	if err == nil {
		return false
	}

	// Extract status code from OpenAI error
	statusCode := ExtractStatusCode(err)
	if statusCode != 0 {
		// Wrap error with status code for base classifier
		wrappedErr := resilience.NewStatusCodeError(statusCode, err)
		return c.baseClassifier.ShouldTripCircuit(wrappedErr)
	}

	// Fall back to base classifier for other errors
	return c.baseClassifier.ShouldTripCircuit(err)
}

// ShouldTripCircuit is a convenience function for OpenAI error classification.
// This is useful for legacy code that uses gobreaker directly.
func ShouldTripCircuit(err error) bool {
	classifier := NewErrorClassifier()
	return classifier.ShouldTripCircuit(err)
}
