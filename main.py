from gymplatformer import make

import pygame

if __name__ == "__main__":

    env = make("PlatformerEnv")
    clock = pygame.time.Clock()
    # sets environment
    env.reset()
    # game loop
    while True:
        clock.tick(15)
        env.render()

        key = pygame.key.get_pressed()

        if key[pygame.K_q]:
            if key[pygame.K_z]:
                env.step(2)
            else:
                env.step(0)
        elif key[pygame.K_d]:
            if key[pygame.K_z]:
                env.step(3)
            else:
                env.step(1)
        elif key[pygame.K_z]:
            env.step(4)
        elif key[pygame.K_ESCAPE]:
            break
        else:
            env.step(5)
