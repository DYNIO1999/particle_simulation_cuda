#include "test.h"
#include <cmath>
#include <chrono>
#include <algorithm>

void updateParticlesCPU(float deltaTime) {

    for (int i = 0; i < MAX_PARTICLES; i++) {
        particles[i].position.x += particles[i].speed.x * deltaTime;
        particles[i].position.y += particles[i].speed.y * deltaTime;

        particles[i].position.x = std::clamp(particles[i].position.x, 0.0f, (float)(SCREEN_WIDTH));
        particles[i].position.y = std::clamp(particles[i].position.y, 0.0f, (float)(SCREEN_HEIGHT));
        float angleChange = 1.0f;
        float angle = std::atan2(particles[i].speed.y, particles[i].speed.x) + angleChange;
        particles[i].speed.x = std::cos(angle);
        particles[i].speed.y = std::sin(angle);

        particles[i].speed.x += 1000.0f * std::cos(angle) * deltaTime;
        particles[i].speed.y += 1000.0f * std::sin(angle) * deltaTime;
    }
}


int main() {

    bool isParticleGpuOn = false;
    bool isParticleCPUOn = true;

    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT,"Particle Simulation");

    for (int i = 0; i < MAX_PARTICLES; i++) {
        particles[i].position = (Vector2){ float(GetRandomValue(0, SCREEN_WIDTH)), float(GetRandomValue(0, SCREEN_HEIGHT)) };
        particles[i].speed = (Vector2){ float(GetRandomValue(-MAX_SPEED, MAX_SPEED)), float(GetRandomValue(-MAX_SPEED, MAX_SPEED))};
        particles[i].color = (Color){
                (unsigned char)GetRandomValue(50, 255),
                (unsigned char)GetRandomValue(50, 255),
                (unsigned char)GetRandomValue(50, 255),
                255
        };

    }

    while (!WindowShouldClose()) {

        float deltatime  = GetFrameTime();

        BeginDrawing();

        ClearBackground(RAYWHITE);


        if(IsKeyPressed(KEY_G)){
            isParticleGpuOn = true;
            isParticleCPUOn = false;
        }
        if(IsKeyPressed(KEY_C)) {
            isParticleGpuOn = false;
            isParticleCPUOn = true;
        }


        if(isParticleCPUOn){
            std::chrono::high_resolution_clock::time_point start_time, end_time;

            start_time = std::chrono::high_resolution_clock::now();

            updateParticlesCPU(deltatime);

            end_time = std::chrono::high_resolution_clock::now();

            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

            std::cout << "Elapsed Time on CPU: " << duration.count() << " ms" << std::endl;
        }


        if(isParticleGpuOn){
            updateParitclesGPU(deltatime);

        }

        std::chrono::high_resolution_clock::time_point start_time, end_time;

        start_time = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < MAX_PARTICLES; i++) {

            DrawCircle(particles[i].position.x, particles[i].position.y, 3, particles[i].color);
        }

        end_time = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        std::cout << "Rendering Time: " << duration.count() << " ms" << std::endl;


        EndDrawing();
    }

    CloseWindow();

    return 0;
}