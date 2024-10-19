import "dotenv/config";
import express from "express";
import cors from "cors";
import { ChatOpenAI } from "@langchain/openai";
import { VectorStoreIndex, SimpleDirectoryReader } from "llamaindex";


// Set up express
const app = express();
const port = process.env.PORT || 3000;

// Set up middleware
app.use(cors());
app.use(express.json());

// Get OpenAi keys
const keys = process.env.OPENAI_API_KEY;
if (!keys) {
  throw new Error("No OpenAI API key provided");
}

const model = new ChatOpenAI({
    openAIApiKey: keys,
    model: "o1-preview",
});

const systemPrompt = "You are a SQL expert assistant. I have provided context to you, which is a schema showing available tables (in the 'name' property) and available columns (in the 'columns) property. The 'description' property explains what data is available on the table. The user is going to ask you a question about their devices, and you are going to reference only the provided schema to determine which tables and columns you need to query in order to answer the question. Output only SQL. The user will run the SQL on their own against a database that matches the schema you have been provided. Never use columns or tables that are not available in the schema. Always return SQL. Never return a column or table that does not exist in schema. Do not try to be helpful, it is more important to be accurate to the schema.";

const formatPrompt = `When generating the SQL:
1. Please do not use the SQL "AS" operator, nor alias tables.  Always reference tables by their full name.
2. If this question is related to an application or program, consider using LIKE instead of something verbatim.
3. If this question is not possible to ask given the osquery schema for a particular operating system, then use empty string.
4. If this question is a "yes" or "no" question, then build the query such that a "yes" returns exactly one row and a "no" returns zero rows.  In other words, if this question is about finding out which hosts match a "yes" or "no" question, then if a host does not match, do not include any rows for it.
5. For each table that you use, only use columns that are documented for that table, and use them as documented.
6. Use only tables that are supported for each target platform, as documented in the schema, considering the examples if they exist, and the available columns.
Please give me all of the above in JSON, with this data shape:
{
  macOSQuery: 'SQL HERE',
  windowsQuery: 'SQL HERE',
  linuxQuery: 'SQL HERE',
  chromeOSQuery: 'SQL HERE'
}
The text 'SQL HERE' is where you will put the SQL necessary to query that type of operating system in osquery. If the data is not avilable in the schema, leave the property empty.
In the resulting JSON report:
1. Never use newline characters within double quotes, and ensure the result is valid JSON.
2. Please do not add any text outside of the JSON report, nor wrap it in a code fence.
3. Ensure your response is valid JSON.`;

const index = await initializeIndex();

async function initializeIndex() {
    const documents = await new SimpleDirectoryReader().loadData({
        directoryPath: "./data",
    });
    return await VectorStoreIndex.fromDocuments(documents);
}

const queryEngine = index.asQueryEngine({
    llm: model,
    temperature: 0.5,
});

app.post("/query", async (req, res) => {
  try {
    
    // Expecting a query in the JSON body
    const query = "System instructions: " + systemPrompt + "\n\n" + "Format instrutions: " + formatPrompt + "\n\n" + "User question: \n\n" + req.body.query;
    if (!query) {
      return res.status(400).send({ error: "Query not provided" });
    }
    
    // Send request and capture response
    const response = await queryEngine.query({ query });

    // Step 1: Access the 'response' string
    const responseString = response;

    // Step 2: Parse the string as JSON to unescape it
    const jsonResponse = JSON.parse(responseString);

    res.send(jsonResponse);

  } catch (error) {
    console.error(error);
    res
      .status(500)
      .send({ error: "An error occurred while processing the query." });
  }
});

app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});