OUTPUT_ARCH( "riscv" )

/* initialize helper variables */
pROM_SIZE = 131072;
pROM_OFFSET = 0;
RAMSIZE = 32768;
RAM_OFFSET = 300500;


/* Stackpointer is at the end of the RAM */
STACK_TOP = ( RAMSIZE + RAM_OFFSET - 4 );

MEMORY {
prom (rx) : ORIGIN = pROM_OFFSET, LENGTH = pROM_SIZE /* size of Instruction Memory */
ram (!rx) : ORIGIN = RAM_OFFSET, LENGTH = RAMSIZE /* size of Data Memory and offset */
}

SECTIONS {
.text :
{
*(.text*)
. = ALIGN(4);
} > prom /* executable data */

.rodata :
{
*(.rodata*);
. = ALIGN(4);
} > ram AT > prom /* read only data */

.sdata :
{
*(.sdata*);
. = ALIGN(4);
} > ram /* global/static variables */

.data :
{
*(.data*);
. = ALIGN(4);
} > ram /* global/static variables */

.bss (NOLOAD) :
{
*(.bss*) FILL(0x00);
*(.sbss*) FILL(0x00);
} > ram /* uninitialized data */

}