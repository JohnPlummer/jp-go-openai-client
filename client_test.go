package openaiclient_test

import (
	"context"
	"errors"
	"testing"

	openaiclient "github.com/JohnPlummer/jp-go-openai-client"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/sashabaranov/go-openai"
)

func TestOpenAIClient(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "OpenAI Client Suite")
}

// mockClient is a test mock for the Client interface.
type mockClient struct {
	response openai.ChatCompletionResponse
	err      error
}

func (m *mockClient) CreateChatCompletion(_ context.Context, _ openai.ChatCompletionRequest) (openai.ChatCompletionResponse, error) {
	return m.response, m.err
}

var _ = Describe("Client", func() {
	Describe("New", func() {
		It("creates a client with valid API key", func() {
			client := openaiclient.New("test-api-key")
			Expect(client).NotTo(BeNil())
		})
	})
})

var _ = Describe("ResilientAdapter", func() {
	var (
		mock    *mockClient
		adapter *openaiclient.ResilientAdapter
	)

	BeforeEach(func() {
		mock = &mockClient{}
		adapter = openaiclient.NewResilientAdapter(mock).(*openaiclient.ResilientAdapter)
	})

	Describe("Execute", func() {
		It("delegates to the wrapped client", func() {
			expectedResponse := openai.ChatCompletionResponse{
				ID: "test-id",
				Choices: []openai.ChatCompletionChoice{
					{Message: openai.ChatCompletionMessage{Content: "Hello!"}},
				},
			}
			mock.response = expectedResponse

			req := openai.ChatCompletionRequest{
				Model: openai.GPT4,
				Messages: []openai.ChatCompletionMessage{
					{Role: openai.ChatMessageRoleUser, Content: "Hi"},
				},
			}

			resp, err := adapter.Execute(context.Background(), req)

			Expect(err).NotTo(HaveOccurred())
			Expect(resp.ID).To(Equal("test-id"))
			Expect(resp.Choices).To(HaveLen(1))
			Expect(resp.Choices[0].Message.Content).To(Equal("Hello!"))
		})

		It("returns errors from the wrapped client", func() {
			mock.err = errors.New("api error")

			req := openai.ChatCompletionRequest{}
			_, err := adapter.Execute(context.Background(), req)

			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(Equal("api error"))
		})
	})
})

var _ = Describe("ResilientClientAdapter", func() {
	It("wraps a ResilientClient as a Client", func() {
		mock := &mockClient{
			response: openai.ChatCompletionResponse{ID: "test"},
		}
		resilientAdapter := openaiclient.NewResilientAdapter(mock)
		clientAdapter := openaiclient.NewResilientClientAdapter(resilientAdapter)

		req := openai.ChatCompletionRequest{}
		resp, err := clientAdapter.CreateChatCompletion(context.Background(), req)

		Expect(err).NotTo(HaveOccurred())
		Expect(resp.ID).To(Equal("test"))
	})
})

var _ = Describe("ErrorClassifier", func() {
	var classifier *openaiclient.ErrorClassifier

	BeforeEach(func() {
		classifier = openaiclient.NewErrorClassifier()
	})

	Describe("ExtractStatusCode", func() {
		It("returns 0 for nil error", func() {
			code := openaiclient.ExtractStatusCode(nil)
			Expect(code).To(Equal(0))
		})

		It("returns 0 for non-OpenAI error", func() {
			err := errors.New("generic error")
			code := openaiclient.ExtractStatusCode(err)
			Expect(code).To(Equal(0))
		})

		It("extracts status code from OpenAI API error", func() {
			apiErr := &openai.APIError{
				HTTPStatusCode: 429,
				Message:        "Rate limit exceeded",
			}
			code := openaiclient.ExtractStatusCode(apiErr)
			Expect(code).To(Equal(429))
		})

		It("extracts status code from wrapped OpenAI API error", func() {
			apiErr := &openai.APIError{
				HTTPStatusCode: 500,
				Message:        "Internal server error",
			}
			wrappedErr := errors.Join(errors.New("context"), apiErr)
			code := openaiclient.ExtractStatusCode(wrappedErr)
			Expect(code).To(Equal(500))
		})
	})

	Describe("IsRetryable", func() {
		It("returns false for nil error", func() {
			Expect(classifier.IsRetryable(nil)).To(BeFalse())
		})

		It("returns true for 429 rate limit error", func() {
			apiErr := &openai.APIError{HTTPStatusCode: 429}
			Expect(classifier.IsRetryable(apiErr)).To(BeTrue())
		})

		It("returns true for 500 server error", func() {
			apiErr := &openai.APIError{HTTPStatusCode: 500}
			Expect(classifier.IsRetryable(apiErr)).To(BeTrue())
		})

		It("returns true for 502 bad gateway", func() {
			apiErr := &openai.APIError{HTTPStatusCode: 502}
			Expect(classifier.IsRetryable(apiErr)).To(BeTrue())
		})

		It("returns true for 503 service unavailable", func() {
			apiErr := &openai.APIError{HTTPStatusCode: 503}
			Expect(classifier.IsRetryable(apiErr)).To(BeTrue())
		})

		It("returns true for 504 gateway timeout", func() {
			apiErr := &openai.APIError{HTTPStatusCode: 504}
			Expect(classifier.IsRetryable(apiErr)).To(BeTrue())
		})

		It("returns false for 400 bad request", func() {
			apiErr := &openai.APIError{HTTPStatusCode: 400}
			Expect(classifier.IsRetryable(apiErr)).To(BeFalse())
		})

		It("returns false for 401 unauthorized", func() {
			apiErr := &openai.APIError{HTTPStatusCode: 401}
			Expect(classifier.IsRetryable(apiErr)).To(BeFalse())
		})

		It("returns false for 403 forbidden", func() {
			apiErr := &openai.APIError{HTTPStatusCode: 403}
			Expect(classifier.IsRetryable(apiErr)).To(BeFalse())
		})

		It("returns false for 404 not found", func() {
			apiErr := &openai.APIError{HTTPStatusCode: 404}
			Expect(classifier.IsRetryable(apiErr)).To(BeFalse())
		})
	})

	Describe("ShouldTripCircuit", func() {
		It("returns false for nil error", func() {
			Expect(classifier.ShouldTripCircuit(nil)).To(BeFalse())
		})

		It("returns true for 401 unauthorized", func() {
			apiErr := &openai.APIError{HTTPStatusCode: 401}
			Expect(classifier.ShouldTripCircuit(apiErr)).To(BeTrue())
		})

		It("returns true for 403 forbidden", func() {
			apiErr := &openai.APIError{HTTPStatusCode: 403}
			Expect(classifier.ShouldTripCircuit(apiErr)).To(BeTrue())
		})

		It("returns false for 429 rate limit", func() {
			apiErr := &openai.APIError{HTTPStatusCode: 429}
			Expect(classifier.ShouldTripCircuit(apiErr)).To(BeFalse())
		})

		It("returns true for 500 server error", func() {
			// Server errors trip the circuit to protect the system from cascading failures
			apiErr := &openai.APIError{HTTPStatusCode: 500}
			Expect(classifier.ShouldTripCircuit(apiErr)).To(BeTrue())
		})

		It("returns true for 503 service unavailable", func() {
			// Service unavailable trips the circuit to avoid hammering a struggling service
			apiErr := &openai.APIError{HTTPStatusCode: 503}
			Expect(classifier.ShouldTripCircuit(apiErr)).To(BeTrue())
		})
	})
})

var _ = Describe("ShouldTripCircuit convenience function", func() {
	It("returns false for nil error", func() {
		Expect(openaiclient.ShouldTripCircuit(nil)).To(BeFalse())
	})

	It("returns true for auth errors", func() {
		apiErr := &openai.APIError{HTTPStatusCode: 401}
		Expect(openaiclient.ShouldTripCircuit(apiErr)).To(BeTrue())
	})

	It("returns false for rate limit errors", func() {
		apiErr := &openai.APIError{HTTPStatusCode: 429}
		Expect(openaiclient.ShouldTripCircuit(apiErr)).To(BeFalse())
	})
})
