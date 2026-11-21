module class1_tree16(input wire [467:0] i, output wire [0:0] o);

reg [0:0] o_reg;
assign o = o_reg;

always @(*) begin
	if((i[144] & (~i[184])) | (i[187] & (~i[77]) & i[184]) | (i[188] & (~i[144]) & (~i[184])) | ((~i[229]) & (~i[188]) & (~i[144]) & (~i[184])) | ((~i[61]) & i[320] & i[143] & i[77] & i[184]) | (i[44] & (~i[320]) & i[143] & i[77] & i[184]) | (i[120] & i[420] & (~i[143]) & i[77] & i[184]))
		o_reg = 0;
	else
		o_reg = 1;
end

endmodule
