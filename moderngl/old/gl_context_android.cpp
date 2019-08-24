#include "gl_context.hpp"

bool GLContext::load(bool standalone) {
    this->standalone = standalone;

    if (standalone) {
    }

    this->window = 0;
    this->display = 0;
    this->context = 0;

    this->error = "android is not yet supported";
    return false;
}

void GLContext::enter() {
}

void GLContext::exit() {
}

bool GLContext::active() {
}
