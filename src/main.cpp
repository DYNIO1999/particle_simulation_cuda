#include <iostream>
#include <cuda_runtime.h>
#include "raylib.h"
#include "test.h"

int main(){
    // Initialization
    //--------------------------------------------------------------------------------------
    const int screenWidth = 800;
    const int screenHeight = 450;

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);



    std::cout<<deviceCount<<'\n';

    char numberAsChar = ('0'+deviceCount);
    char* text = new char[2];
    text[0]= numberAsChar;
    text[1]= '\0';


    float* vecA = new float[3];
    float* vecB = new float[3];
    float *d_x, *d_y;

    for(size_t i =0; i<3;i++){
        vecA[i] = (0.1*(float)i);
        vecB[i] = vecA[i];
    }


    //performTest();

    InitWindow(screenWidth, screenHeight, "raylib [core] example - basic window");

    SetTargetFPS(60);               // Set our game to run at 60 frames-per-second
    //--------------------------------------------------------------------------------------

    // Main game loop
    while (!WindowShouldClose())    // Detect window close button or ESC key
    {
        // Update
        //----------------------------------------------------------------------------------
        // TODO: Update your variables here
        //----------------------------------------------------------------------------------

        // Draw
        //----------------------------------------------------------------------------------
        BeginDrawing();

        ClearBackground(RAYWHITE);

        DrawText(text, screenWidth/2, 200, 100, BLUE);

        EndDrawing();
        //----------------------------------------------------------------------------------
    }

    // De-Initialization
    //--------------------------------------------------------------------------------------
    CloseWindow();        // Close window and OpenGL context
    //--------------------------------------------------------------------------------------

    delete[] text;

    return 0;
}
