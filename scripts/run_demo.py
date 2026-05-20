from camera_reconstruction.demo import _parse_args, main


if __name__ == "__main__":
    args = _parse_args()
    main(out_dir=args.out_dir)
