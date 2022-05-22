#include "Interpreter.hpp"

#include <iostream>

Interpreter::~Interpreter() {
    for (auto command : commands)
        delete command;
}

int Interpreter::addCommand(const char* filename) {
    FILE *f = fopen(filename, "r");
    if (!f)
        return -1;
    
    size_t size;
    fseek(f, 0, SEEK_END);
    size = ftell(f);
    
    char* command = new char [size];
    if (!command)
        return -1;
    
    fseek(f, 0, SEEK_SET);
    fread(command, 1, size, f);
    fclose(f);
    
    Interpreter::addCommand(command, 0);
    
    delete [] command;
    
    return 0;
}

int Interpreter::addCommand(const char* command, int) {
    curToken = UNKNOWN;
    ss = std::stringstream(command);
    
    InterpreteNode *interpreter = new InterpreteNode;
    
    while (ss) {
        statement(*interpreter);
    }
    
    commands.push_back(interpreter);
    
    return 0;
}

int Interpreter::printCommand(void) {
    for (int i = 0; i < commands.size(); ++i) {
        printf("Command %d:\n", i + 1);
        commands[i]->print();
    }
    
    return 0;
}

int Interpreter::doCommand() {
    int success = 0;
    for (int i = 0; i < commands.size() && success == 0; ++i) {
        doCommand(i);
        success = getTable("success");
    }
    return success;
}

int Interpreter::doCommand(int index) {
    auto &interpreter = commands[index];
    
    for (int i = 0; i < interpreter->command_line.size(); ) {
        auto &step_command = interpreter->command_line[i];
        
        if (step_command->type == CommandType::IF) {
            int judge = evaluateTree(step_command->command);
            if (!judge)
                i += step_command->step;
            else
                i += 1;
        } else {
            evaluateTree(step_command->command);
            i += step_command->step;
        }
    }
    
    return 0;
}

void Interpreter::addTable(std::string name, float value) {
    table[name] = value;
}

float Interpreter::getTable(std::string name) {
    if (table.find(name) != table.end())
        return table[name];
    printf("Invalid get %s\n", name.c_str());
    return 0;
}

void Interpreter::printTable(void) {
    for (const auto& t : table) {
        std::cout << t.first << ": " << t.second << std::endl;
    }
}

CommandNode* Interpreter::statement(InterpreteNode& interpreter) {
    CommandNode* retp = nullptr;
    
    if (match(ENDFILE)) {
        return retp;
    } else if (match(END)) {
        advance();
    } else if (match(IF)) {
        advance();
        
        if (match(LPAREN))
            advance();
        else
            printf("[If] Miss (\n");
        
        // If condition
        int step_stack = interpreter.step;
        interpreter.step = 1;
        retp = expr();
        auto command = new StepCommand(1, retp, CommandType::IF);
        interpreter.command_line.push_back(command);
        
        if (match(RPAREN))
            advance();
        else
            printf("[If] Miss )\n");
        
        statement(interpreter);
        
        auto if_jump = interpreter.command_line.back();
        int else_step_stack = interpreter.step;
        
        if (match(ELSE)) {
            advance();
            interpreter.step = 0;
            
            statement(interpreter);
            if_jump->step += interpreter.step;
            interpreter.step += else_step_stack;
        }

        command->step = interpreter.step - if_jump->step + 1;
        interpreter.step += step_stack;
    } else if (match(LBRACE)) {
        advance();
        
        while (!match(RBRACE)) {
            statement(interpreter);
        }
        
        if (match(RBRACE))
            advance();
        else
            printf("Miss }\n");
    } else {
        interpreter.step++;
        retp = expr();
        auto command = new StepCommand(1, retp, CommandType::STATEMENT);
        interpreter.command_line.push_back(command);
        if (match(SEMICOLON)) {
            advance();
        } else {
            printf("Synatex error\n");
        }
    }
    return retp;
}

int Interpreter::evaluateTree(CommandNode *root) {
    int retval = 0, lv = 0, rv = 0;

    if (root != NULL) {
        switch (root->data) {
            case ID:
                retval = getval(root->lexeme);
                break;
            case INT:
                retval = atoi(root->lexeme);
                break;
            case ASSIGN:
                rv = evaluateTree(root->right);
                retval = setval(root->left->lexeme, rv);
                break;
            case ADDSUB:
            case MULDIV:
                lv = evaluateTree(root->left);
                rv = evaluateTree(root->right);
                if (strcmp(root->lexeme, "+") == 0) {
                    retval = lv + rv;
                } else if (strcmp(root->lexeme, "-") == 0) {
                    retval = lv - rv;
                } else if (strcmp(root->lexeme, "*") == 0) {
                    retval = lv * rv;
                } else if (strcmp(root->lexeme, "/") == 0) {
                    if (rv == 0)
//                        Error(DIVZERO);
                    retval = lv / rv;
                }
                break;
            case LOGICAL:
                lv = evaluateTree(root->left);
                rv = evaluateTree(root->right);
                if (strcmp(root->lexeme, "==") == 0) {
                    retval = lv == rv;
                } else if (strcmp(root->lexeme, "!=") == 0) {
                    retval = lv != rv;
                } else if (strcmp(root->lexeme, ">") == 0) {
                    retval = lv > rv;
                } else if (strcmp(root->lexeme, "<") == 0) {
                    retval = lv < rv;
                } else if (strcmp(root->lexeme, ">=") == 0) {
                    retval = lv >= rv;
                } else if (strcmp(root->lexeme, "<=") == 0) {
                    retval = lv <= rv;
                } else if (strcmp(root->lexeme, "&&") == 0) {
                    retval = lv && rv;
                } else if (strcmp(root->lexeme, "||") == 0) {
                    retval = lv || rv;
                }
                break;
            default:
                retval = 0;
        }
    }
    return retval;
}

int Interpreter::getval(char *str) {
    if (table.find(str) != table.end()) {
        return table[str];
    }
    printf("Not found %s in memory\n", str);
    
    return 0;
}

int Interpreter::setval(char *str, int val) {
    if (table.find(str) != table.end()) {
        table[str] = val;
        return val;
    }
    
    table[str] = val;
    
    return val;
}

void Interpreter::printPrefix(CommandNode *root) {
    if (root != NULL) {
        printf("%s ", root->lexeme);
        printPrefix(root->left);
        printPrefix(root->right);
    }
}

CommandNode* Interpreter::factor(void) {
    CommandNode* retp = NULL, *left = NULL;

    if (match(INT)) {
        retp = new CommandNode(INT, getLexeme());
        advance();
    } else if (match(ID)) {
        left = new CommandNode(ID, getLexeme());
        advance();
        if (!match(ASSIGN)) {
            retp = left;
        } else {
            retp = new CommandNode(ASSIGN, getLexeme());
            advance();
            retp->left = left;
            retp->right = expr();
        }
    } else if (match(ADDSUB)) {
        retp = new CommandNode(ADDSUB, getLexeme());
        retp->left = new CommandNode(INT, "0");
        advance();
        if (match(INT)) {
            retp->right = new CommandNode(INT, getLexeme());
            advance();
        } else if (match(ID)) {
            retp->right = new CommandNode(ID, getLexeme());
            advance();
        } else if (match(LPAREN)) {
            advance();
            retp->right = expr();
            if (match(RPAREN))
                advance();
            else
                printf("MISPAREN\n");
        } else {
            printf("NOTNUMID\n");
        }
    } else if (match(LPAREN)) {
        advance();
        retp = expr();
        if (match(RPAREN))
            advance();
        else
            printf("MISPAREN\n");
    } else {
        std::cout << curToken << ": " << getLexeme() << std::endl;
        printf("NOTNUMID\n");
    }
    return retp;
}

CommandNode* Interpreter::term(void) {
    CommandNode* node = factor();
    return term_tail(node);
}

CommandNode* Interpreter::term_tail(CommandNode* left) {
    CommandNode* node = NULL;

    if (match(MULDIV)) {
        node = new CommandNode(MULDIV, getLexeme());
        advance();
        node->left = left;
        node->right = factor();
        return term_tail(node);
    } else if (match(LOGICAL)) {
        node = new CommandNode(LOGICAL, getLexeme());
        advance();
        node->left = left;
        node->right = factor();
        return term_tail(node);
    } else {
        return left;
    }
}

CommandNode* Interpreter::expr(void) {
    CommandNode* node = term();
    return expr_tail(node);
}

CommandNode* Interpreter::expr_tail(CommandNode* left) {
    CommandNode* node = NULL;

    if (match(ADDSUB)) {
        node = new CommandNode(ADDSUB, getLexeme());
        advance();
        node->left = left;
        node->right = term();
        return expr_tail(node);
    } else {
        return left;
    }
}

char* Interpreter::getLexeme(void) {
    return lexeme;
}

TokenSet Interpreter::getToken(void) {
    int i;
    char c = '\0';
    
    while ((c = ss.get()) == ' ' || c == '\t');
           
    if (isdigit(c)) {
        lexeme[0] = c;
        c = ss.get();
        i = 1;
        while (isdigit(c) && i < MAXLEN) {
            lexeme[i] = c;
            ++i;
            c = ss.get();
        }
        ss.putback(c);
        lexeme[i] = '\0';
        return INT;
    } else if (c == '+' || c == '-') {
        lexeme[0] = c;
        c = ss.get();
        if (c == lexeme[0]) {
            lexeme[1] = c;
            lexeme[2] = '\0';
            return INCDEC;
        } else {
            ss.putback(c);
            lexeme[1] = '\0';
            return ADDSUB;
        }
    } else if (c == '&' || c == '|' || c == '^') {
        lexeme[0] = c;
        c = ss.get();
        if (lexeme[0] == '&' && c == '&') {
            lexeme[1] = c;
            lexeme[2] = '\0';
            return LOGICAL;
        } else if (lexeme[0] == '|' && c == '|') {
            lexeme[1] = c;
            lexeme[2] = '\0';
            return LOGICAL;
        }
        ss.putback(c);
        lexeme[1] = '\0';
        return LOGICAL;
    } else if (c == '*' || c == '/') {
        lexeme[0] = c;
        lexeme[1] = '\0';
        return MULDIV;
    } else if (c == '\n') {
        lexeme[0] = '\0';
        return END;
    } else if (c == '=') {
        lexeme[0] = c;
        c = ss.get();
        if (c == '=') {
            lexeme[1] = c;
            lexeme[2] = '\0';
            return LOGICAL;
        }
        ss.putback(c);
        lexeme[1] = '\0';
        return ASSIGN;
    } else if (c == '!') {
        lexeme[0] = c;
        c = ss.get();
        if (c == '=') {
            lexeme[1] = c;
            lexeme[2] = '\0';
            return LOGICAL;
        }
        ss.putback(c);
        return UNKNOWN;
    } else if (c == '(') {
        strcpy(lexeme, "(");
        return LPAREN;
    } else if (c == ')') {
        strcpy(lexeme, ")");
        return RPAREN;
    } else if (c == '{') {
        strcpy(lexeme, "{");
        return LBRACE;
    } else if (c == '}') {
        strcpy(lexeme, "}");
        return RBRACE;
    } else if (c == ';') {
        strcpy(lexeme, ";");
        return SEMICOLON;
    } else if (c == '>') {
        lexeme[0] = c;
        c = ss.get();
        if (c == '=') {
            lexeme[1] = c;
            lexeme[2] = '\0';
            return LOGICAL;
        }
        ss.putback(c);
        lexeme[1] = '\0';
        return LOGICAL;
    } else if (c == '<') {
        lexeme[0] = c;
        c = ss.get();
        if (c == '=') {
            lexeme[1] = c;
            lexeme[2] = '\0';
            return LOGICAL;
        }
        ss.putback(c);
        lexeme[1] = '\0';
        return LOGICAL;
    } else if (isalpha(c) || c == '_') {
        lexeme[0] = c;
        c = ss.get();
        i = 1;
        while (isalpha(c) || isdigit(c) || c == '_') {
            lexeme[i] = c;
            ++i;
            c = ss.get();
        }
        ss.putback(c);
        lexeme[i] = '\0';
        if (i > 1) {
            if (strcmp(lexeme, "if") == 0)
                return IF;
            else if (strcmp(lexeme, "else") == 0) {
                return ELSE;
            }
        }
        return ID;
    } else if (c == EOF) {
        return ENDFILE;
    } else {
        return UNKNOWN;
    }
    return UNKNOWN;
}

void Interpreter::advance(void) {
    curToken = getToken();
}

int Interpreter::match(TokenSet token) {
    if (curToken == UNKNOWN)
        advance();
    return token == curToken;
}

std::ostream& operator<<(std::ostream& o, TokenSet t) {
    switch (t) {
        case UNKNOWN:
            return o << "UNKNOWN";
        case END:
            return o << "END";
        case ENDFILE:
            return o << "ENDFILE";
        case INT:
            return o << "INT";
        case ID:
            return o << "ID";
        case ADDSUB:
            return o << "ADDSUB";
        case MULDIV:
            return o << "MULDIV";
        case LOGICAL:
            return o << "LOGICAL";
        case INCDEC:
            return o << "INCDEC";
        case ASSIGN:
            return o << "ASSIGN";
        case LPAREN:
            return o << "LPAREN";
        case RPAREN:
            return o << "RPAREN";
        case IF:
            return o << "IF";
        case ELSE:
            return o << "ELSE";
        default:
            return o << "UNDEFINED";
    }
    return o;
}

void StepCommand::freeTree(CommandNode* root) {
//    if (root) {
//        delete root->left;
//        delete root->right;
//        delete root;
//    }
}

void InterpreteNode::print() {
    for (int i = 0; i < command_line.size(); ++i) {
        printf("%s: ", (command_line[i]->type == CommandType::IF) ? "If" : "  ");
        printPrefix(command_line[i]->command);
        printf(" -> step: %d\n", command_line[i]->step);
    }
}

void InterpreteNode::printPrefix(CommandNode *root) {
    if (root != NULL) {
        printf("%s ", root->lexeme);
        printPrefix(root->left);
        printPrefix(root->right);
    }
}
