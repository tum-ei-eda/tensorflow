#include<stdint.h>
#include<string.h>

//initialize Stackpointer;
__asm__ ("la sp, STACK_TOP");

//pre declaration of functions
int main();
void start();


void start(void){

    int i;

    i = main();
    //idle binary when main is complete
    while(!i){}
}