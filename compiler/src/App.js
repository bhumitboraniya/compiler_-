import React, { useState, useEffect } from "react";
import axios from "axios";
import "./App.css";
import { Modal, Button, Spinner } from "react-bootstrap";
import { ChevronDown, Zap, Code2, AlertCircle, Terminal, X, Monitor } from "lucide-react";
import ReactJson from 'react-json-view';
function App() {
  const [source, setSource] = useState("");
  const [tokens, setTokens] = useState([]);
  const [ast, setAst] = useState([]);
  const [errors, setErrors] = useState([]);
  const [nasmCode, setNasmCode] = useState("");
  const [currentPhase, setCurrentPhase] = useState(0); // Track the current phase (1 to 5)
  const [loading, setLoading] = useState(false); // Track loading state
  const [modalInfo, setModalInfo] = useState({
    show: false,
    title: "",
    content: "",
  });
  const [simulationOutput, setSimulationOutput] = useState("");

  const phases = ["Tokenize", "Parse", "Type Check", "Output"];

  // Add icons mapping
  const phaseIcons = {
    "Tokenize": <Zap className="w-4 h-4" />,
    "Parse": <Code2 className="w-4 h-4" />,
    "Type Check": <AlertCircle className="w-4 h-4" />,
    "Generate MASM Code": <Terminal className="w-4 h-4" />,
    "Simulate Execution": <Monitor className="w-4 h-4" />,
    "Output": <Terminal className="w-4 h-4" />
  };

  const handleCompile = async () => {
    setLoading(true);
    setCurrentPhase(1); // Start with Tokenize phase

    try {
      const tokenizeResponse = await axios.post(
        "http://127.0.0.1:5000/tokenize",
        { source }
      );
      setTokens(tokenizeResponse?.data?.tokens);
    } catch (error) {
      handlePhaseError("Tokenize", error);
    }
  };

  useEffect(() => {
    if (currentPhase === 1 && tokens.length > 0) {
      // Phase 2: Parse
      const parseTokens = async () => {
        setCurrentPhase(2);
        try {
          const parseResponse = await axios.post(
            "http://127.0.0.1:5000/parse",
            { tokens }
          );
          setAst(parseResponse.data.ast);
        } catch (error) {
          handlePhaseError("Parse", error);
        }
      };
      parseTokens();
    }
  }, [tokens]);

  useEffect(() => {
    if (currentPhase === 2 && ast.length > 0) {
      // Phase 3: Type Check
      const checkTypes = async () => {
        setCurrentPhase(3);
        try {
          const typeCheckResponse = await axios.post(
            "http://127.0.0.1:5000/typecheck",
            { ast }
          );
          setErrors(typeCheckResponse.data.errors);

          // Display type check errors in the modal
          if (typeCheckResponse.data.errors.length > 0) {
            handleOpenModal(
              "Type Check Error",
              `Errors: ${JSON.stringify(
                typeCheckResponse.data.errors,
                null,
                2
              )}`
            );
            setLoading(false); // Stop loading if there are errors
            return; // Stop further processing
          }
        } catch (error) {
          handlePhaseError("Type Check", error);
        }
      };
      checkTypes();
    }
  }, [ast]);

  useEffect(() => {
    if (currentPhase === 3 && errors.length === 0) {
      // Phase 4: Generate MASM Code
      const generateCode = async () => {
        setCurrentPhase(4);
        try {
          const generateResponse = await axios.post(
            "http://127.0.0.1:5000/generate",
            { ast }
          );
          
          if (generateResponse.data.error) {
            handlePhaseError("Runtime", generateResponse.data.error);
            return;
          }
          
          setNasmCode(generateResponse.data.nasm_code);
          setSimulationOutput(generateResponse.data.simulation.output);
          setCurrentPhase(5);
          setLoading(false);
        } catch (error) {
          handlePhaseError("Generate/Runtime", error);
        }
      };
      generateCode();
    }
  }, [errors]);

  const handleReset = () => {
    setSource("");
    setTokens([]);
    setAst([]);
    setErrors([]);
    setNasmCode("");
    setCurrentPhase(0);
    setLoading(false); // Ensure loading is also reset
    setSimulationOutput("");
  };

  const handlePhaseError = (phase, error) => {
    const errorMessage =
      error.response?.data?.error || error.message || "Unknown error";
    setModalInfo({
      show: true,
      title: `${phase} Error`,
      content: `An error occurred in the ${phase} phase:\n\n${errorMessage}`,
    });
    setCurrentPhase(0); // Reset phase on error
    setLoading(false); // Stop loading if error
  };

  const handleOpenModal = (title, content) => {
    setModalInfo({ show: true, title, content });
  };

  const handleCloseModal = () => {
    setModalInfo({ show: false, title: "", content: "" });
  };
  


  function getPhaseDetails(phase) {
    switch (phase) {
      case "Tokenize":
        return `Tokens: ${JSON.stringify(tokens, null, 2)}`;
      case "Parse":
        return `AST: ${JSON.stringify(ast, null, 2)}`;
      case "Type Check":
        return `Errors: ${JSON.stringify(errors, null, 2)}`;
      case "Generate MASM Code":
        return `MASM Code: ${nasmCode}`;
      case "Output":
        const output = simulationOutput || "No output generated";
        // Format the output to show floating point numbers with precision
        const formattedOutput = output.replace(/(\d+\.\d+)/g, num => 
          parseFloat(num).toFixed(6) // Changed from toExponential to toFixed
        );
        return `Program Output:\n${formattedOutput}`;
      default:
        return "";
    }
  }

  // Update examples object with more comprehensive examples
  const examples = {
    arithmetic: {
      name: "Basic Arithmetic",
      code: `INSERT 10
INSERT 5
ADD      // 10 + 5 = 15
PRINT "Result after addition:"
PRINT "Final result is"
INSERT 3
MUL      // 15 * 3 = 45
PRINT "Result after multiplication:"
PRINT "Final result is"
INSERT 5
DIV      // 45 / 5 = 9
PRINT "Result after division:"
PRINT "Final result is"
EXIT`
    },
    complex: {
      name: "Complex Math",
      code: `INSERT 100
INSERT 20
DIV        // 100 / 20 = 5
INSERT 3
MUL        // 5 * 3 = 15
INSERT 25
INSERT 5
DIV        // 25 / 5 = 5
ADD        // 15 + 5 = 20
PRINT "Complex calculation result:"
PRINT "Final result is"
EXIT`
    },
    error: {
      name: "Division by Zero",
      code: `INSERT 20
INSERT 0
DIV        // Error: Division by zero
PRINT "This won't execute"
EXIT`
    }
  };

  return (
    <div className="container mt-5">
      <h1 className="text-center">Compiler Construction</h1>

      {/* Input source code */}
      <textarea
        className="form-control shadow-sm"
        rows="10"
        value={source}
        onChange={(e) => setSource(e.target.value)}
        placeholder="Enter your code here..."
      />

      {/* Vertical Phase List */}
      <div className="main-container max-w-md mx-auto space-y-4 my-8 flex flex-col">
        {phases.map((phase, index) => (
          <div key={index} className="relative">
            <div
              className={`
                group flex items-center gap-3 p-4 rounded-lg
                transition-all duration-300 ease-in-out
                ${currentPhase > index 
                  ? "bg-primary/10 hover:bg-primary/20 cursor-pointer" 
                  : "bg-gray-50"}
                ${currentPhase === index 
                  ? "animate-pulse border-l-4 border-primary" 
                  : "border-l-4 border-transparent"}
              `}
              onClick={() => {
                if (currentPhase > index) {
                  const modalContent = getPhaseDetails(phase);
                  handleOpenModal(phase, modalContent);
                }
              }}
            >
              <div className={`
                p-2 rounded-full 
                ${currentPhase > index 
                  ? "bg-primary/20 text-primary" 
                  : "bg-gray-100 text-gray-400"}
                transition-all duration-300 group-hover:scale-110
              `}>
                {phaseIcons[phase]}
              </div>
              
              <div className="flex-1">
                <h3 className={`
                  font-medium text-sm
                  ${currentPhase > index ? "text-primary" : "text-gray-600"}
                `}>
                  {phase}
                </h3>
                {currentPhase > index && (
                  <p className="text-xs text-gray-500 mt-1">
                    Click to view details
                  </p>
                )}
              </div>

              {index < phases.length - 1 && (
                <ChevronDown 
                  className={`
                    w-5 h-5 absolute -bottom-4 left-5 z-10
                    transition-all duration-300
                    ${currentPhase > index 
                      ? "text-primary" 
                      : "text-gray-300"}
                  `}
                />
              )}
            </div>
            
            {index < phases.length - 1 && (
              <div className={`
                w-px h-4 ml-[1.3rem] bg-gray-200
                ${currentPhase > index ? "bg-primary/30" : ""}
              `} />
            )}
          </div>
        ))}
      </div>

      {/* Compile and Reset Buttons */}
      <div className="text-center mt-3">
        <button
          className="btn btn-primary mx-2"
          onClick={handleCompile}
          disabled={loading || currentPhase !== 0}
        >
          {loading ? (
            <Spinner as="span" animation="border" size="sm" />
          ) : (
            "Compile"
          )}
        </button>
        <button
          className="btn btn-secondary mx-2"
          onClick={handleReset}
          disabled={loading}
        >
          Reset
        </button>
      </div>

      {/* Replace Examples Button with Examples Section */}
      <div className="examples-section mt-4">
        <h4 className="text-center mb-3">Example Programs</h4>
        <div className="d-flex justify-content-center gap-2">
          {Object.entries(examples).map(([key, example]) => (
            <button
              key={key}
              className="btn btn-info mx-2"
              onClick={() => setSource(example.code)}
            >
              {example.name}
            </button>
          ))}
        </div>
        <div className="text-center mt-2">
          <small className="text-muted">
            Click an example to load it into the editor
          </small>
        </div>
      </div>

      {/* Bootstrap Modal for detailed view */}
      <Modal show={modalInfo.show} onHide={handleCloseModal}>
        <Modal.Header closeButton>
          <Modal.Title>{modalInfo.title}</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <pre>{modalInfo.content}</pre>
        </Modal.Body>
        <Modal.Footer>
          <Button variant="secondary" onClick={handleCloseModal}>
            Close
          </Button>
        </Modal.Footer>
      </Modal>
    </div>
  );
}

export default App;
