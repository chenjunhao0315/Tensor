#ifndef Interpreter_hpp
#define Interpreter_hpp

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <sstream>
#include <vector>
#include <stack>
#include <tuple>
#include <unordered_map>

namespace otter {
namespace core {

#define MAXLEN 256

typedef enum {
    UNKNOWN, END, ENDFILE,
    INT, ID, FLOAT,
    ADDSUB, MULDIV, LOGICAL,
    INCDEC, ASSIGN,
    LPAREN, RPAREN,
    IF, ELSE,
    LBRACE, RBRACE,
    SEMICOLON
} TokenSet;

std::ostream& operator<<(std::ostream& s, TokenSet t);

typedef struct {
    int val;
    char name[MAXLEN];
} Symbol;

class CommandNode;
class InterpreteNode;

class Interpreter {
public:
    ~Interpreter();
    Interpreter(const Interpreter&) = delete;
    Interpreter(Interpreter&&) = delete;
    Interpreter() : curToken(UNKNOWN) {}
    
    int addCommand(const char* command, int);
    int addCommand(const char* filename);

    int printCommand(void);
    
    int doCommand();
    int doCommand(int index);
    
    void addTable(std::string name, float value);
    float getTable(std::string name);
    void printTable(void);
    
private:
    std::stringstream ss;
    
    TokenSet curToken;
    char lexeme[MAXLEN];
    
    int status;
    
    CommandNode* statement(InterpreteNode& interpreter);
    
    void advance(void);
    int match(TokenSet token);
    TokenSet getToken(void);
    char* getLexeme(void);
    
    CommandNode* expr(void);
    CommandNode* expr_tail(CommandNode* left);
    CommandNode* term(void);
    CommandNode* term_tail(CommandNode* left);
    CommandNode* factor(void);
    
    void printPrefix(CommandNode *root);
    
    float evaluateTree(CommandNode *root);
    float getval(char *str);
    float setval(char *str, float val);

    std::vector<InterpreteNode*> commands;
    std::unordered_map<std::string, float> table;
};

class CommandNode {
public:
    CommandNode(TokenSet tok, const char *lexe) {
        strcpy(lexeme, lexe);
        data = tok;
        val = 0;
        left = nullptr;
        right = nullptr;
    }
    
    TokenSet data;
    int val;
    char lexeme[MAXLEN];
    CommandNode* left;
    CommandNode* right;
};

enum class CommandType {
    IF,
    STATEMENT
};

struct StepCommand {
    StepCommand(int s, CommandNode* c, CommandType t) : step(s), command(c), type(t) {}
    ~StepCommand() {
        freeTree(command);
    }
    void freeTree(CommandNode* root);
    int step;
    CommandNode* command;
    CommandType type;
};

class InterpreteNode {
public:
    ~InterpreteNode() {
        for (auto command : command_line)
            delete command;
    }
    void print(void);
    void printPrefix(CommandNode *root);

    int step = 1;
    std::vector<StepCommand*> command_line;
};

}   // end namespace core
}   // end namespace otter

#endif
